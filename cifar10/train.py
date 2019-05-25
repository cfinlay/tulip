import random
import time, datetime
import os, shutil
import yaml
import ast, bisect
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import grad
import torchnet as tnt

import dataloader
from dataloader import cutout
import models


# -------------
# Initial setup
# -------------

# Parse command line arguments
from argparser import parser
args = parser.parse_args()


# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# Set random seed
if args.seed is None:
    args.seed = int(time.time())
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Set and create logging directory
if args.logdir is None:
    args.logdir = os.path.join('./logs/','cifar10',args.model,
            '{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))
os.makedirs(args.logdir, exist_ok=True)


# Print arguments to std out
# and save argument values to yaml file
print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

args_file_path = os.path.join(args.logdir, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)


workers=4
test_loader = getattr(dataloader, 'cifar10')(args.datadir,
        mode='test', transform=False,
        batch_size=args.test_batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=has_cuda)

transforms = [cutout(args.cutout,channels=3 )]
train_loader = getattr(dataloader, 'cifar10')(args.datadir,
        mode='train', transform=True,
        batch_size=args.batch_size,
        training_transforms = transforms,
        num_workers=workers,
        shuffle=True,
        pin_memory=has_cuda,
        drop_last=True)


model = getattr(models, args.model)()

criterion = nn.CrossEntropyLoss()
train_criterion = nn.CrossEntropyLoss(reduction='none')

if has_cuda:
    criterion = criterion.cuda(0)
    train_criterion = train_criterion.cuda(0)
    model = model.cuda(0)

optimizer = optim.SGD(model.parameters(),
                  lr = args.lr,
                  weight_decay = args.decay,
                  momentum = args.momentum,
                  nesterov = False)

def scheduler(optimizer,lr_schedule):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler
schedule = scheduler(optimizer,args.lr_schedule)



# --------
# Training
# --------


trainlog = os.path.join(args.logdir,'training.csv')
traincolumns = ['index','time','loss', 'regularizer']
with open(trainlog,'w') as f:
    logger = csv.DictWriter(f, traincolumns)
    logger.writeheader()

ix=0 #count of gradient steps

tik = args.penalty

regularizing = tik>0

h = args.h # finite difference step size

def train(epoch, ttot):
    global ix

    # Put the model in train mode (unfreeze batch norm parameters)
    model.train()

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()

    with open(trainlog,'a') as f:
        logger = csv.DictWriter(f, traincolumns)

        for batch_ix, (x, target) in enumerate(train_loader):

            if has_cuda:
                x = x.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            if regularizing:
                x.requires_grad_(True)

            prediction = model(x)
            lx = train_criterion(prediction, target)
            loss = lx.mean()


            # Compute finite difference approximation of directional derivative of grad loss wrt inputs
            if regularizing:

                dx = grad(loss, x, retain_graph=True)[0]
                sh = dx.shape
                x.requires_grad_(False)

                # v is the finite difference direction.
                # For example, if norm=='L2', v is the gradient of the loss wrt inputs
                v = dx.view(sh[0],-1)
                Nb, Nd = v.shape


                if args.norm=='L2':
                    nv = v.norm(2,dim=-1,keepdim=True)
                    nz = nv.view(-1)>0
                    v[nz] = v[nz].div(nv[nz])
                if args.norm=='L1':
                    v = v.sign()
                    v = v/np.sqrt(Nd)
                elif args.norm=='Linf':
                    vmax, Jmax = v.abs().max(dim=-1)
                    sg = v.sign()
                    I = torch.arange(Nb, device=v.device)
                    sg = sg[I,Jmax]

                    v = torch.zeros_like(v)
                    I = I*Nd
                    Ix = Jmax+I
                    v.put_(Ix, sg)

                v = v.view(sh)
                xf = x + h*v

                mf = model(xf)
                lf = train_criterion(mf,target)
                if args.fd_order=='O2':
                    xb = x - h*v
                    mb = model(xb)
                    lb = train_criterion(mb,target)
                    H = 2*h
                else:
                    H = h
                    lb = lx
                dl = (lf-lb)/H # This is the finite difference approximation
                               # of the directional derivative of the loss


            tik_penalty = torch.tensor(np.nan)
            dlmean = torch.tensor(np.nan)
            dlmax = torch.tensor(np.nan)
            if tik>0:
                dl2 = dl.pow(2)
                tik_penalty = dl2.mean()/2
                loss = loss + tik*tik_penalty

            loss.backward()

            optimizer.step()

            if np.isnan(loss.data.item()):
                raise ValueError('model returned nan during training')

            t = ttot + time.perf_counter() - tepoch
            fmt = '{:.4f}'
            logger.writerow({'index':ix,
                'time': fmt.format(t),
                'loss': fmt.format(loss.item()),
                'regularizer': fmt.format(tik_penalty) })

            if (batch_ix % args.log_interval == 0 and batch_ix > 0):
                print('[%2d, %3d] penalized training loss: %.3g' %
                    (epoch, batch_ix, loss.data.item()))
            ix +=1

    if has_cuda:
        torch.cuda.synchronize()

    return ttot + time.perf_counter() - tepoch


# ------------------
# Evaluate test data
# ------------------
testlog = os.path.join(args.logdir,'test.csv')
testcolumns = ['epoch','time','fval','pct_err','train_fval','train_pct_err']
with open(testlog,'w') as f:
    logger = csv.DictWriter(f, testcolumns)
    logger.writeheader()

def test(epoch, ttot):
    model.eval()

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)


            top1_train.add(output.data, target.data)
            loss = criterion(output, target)
            train_loss.add(loss.data.item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        # Evaluate test data
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)

            loss = criterion(output, target)

            top1.add(output, target)
            test_loss.add(loss.item())

        t1 = top1.value()[0]
        l = test_loss.value()[0]

    # Report results
    with open(testlog,'a') as f:
        logger = csv.DictWriter(f, testcolumns)
        fmt = '{:.4f}'
        logger.writerow({'epoch':epoch,
            'fval':fmt.format(l),
            'pct_err':fmt.format(t1),
            'train_fval':fmt.format(lt),
            'train_pct_err':fmt.format(t1t),
            'time':fmt.format(ttot)})

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('training loss',lt,t1t))

    return test_loss.value()[0], top1.value()[0]




def main():


    save_model_path = os.path.join(args.logdir, 'checkpoint.pth.tar')
    best_model_path = os.path.join(args.logdir, 'best.pth.tar')

    pct_max = 90.
    fail_count = fail_max = 5
    time = 0.
    pct0 = 100.
    for e in range(args.epochs):

        # Update the learning rate
        schedule.step()

        time = train(e, time)

        loss, pct_err= test(e,time)
        if pct_err >= pct_max:
            fail_count -= 1

        torch.save({'ix': ix,
                    'epoch': e + 1,
                    'model': args.model,
                    'state_dict':model.state_dict(),
                    'pct_err': pct_err,
                    'loss': loss
                    }, save_model_path)
        if pct_err < pct0:
            shutil.copyfile(save_model_path, best_model_path)
            pct0 = pct_err

        if fail_count < 1:
            raise ValueError('Percent error has not decreased in %d epochs'%fail_max)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
