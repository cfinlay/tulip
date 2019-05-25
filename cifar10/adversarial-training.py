"""Adversarial training in the style of Madry et al [1]

   [1] Madry et al, "Towards Deep Learning models resistant to adversarial attacks",
       arXiv:1706.06083
"""
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
parser.add_argument('--AT-dt', type=float, default=0.43,
        help='adversarial training step size (default: 0.435)')
parser.add_argument('--AT-steps', type=int, default=7,
        help='number of AT steps (default: 7)')
parser.add_argument('--AT-ball', type=float, default=0.0314,
        help='AT projection ball size (default: 0.0314)')
args = parser.parse_args()

assert args.penalty==0, 'Regularization + AT not implemented'


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
traincolumns = ['index','time','loss']
with open(trainlog,'w') as f:
    logger = csv.DictWriter(f, traincolumns)
    logger.writeheader()

ix=0 #count of gradient steps


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
            x0 = x.clone()

            for p in model.parameters():
                p.requires_grad_(False)

            for k in range(args.AT_steps):
                x.requires_grad_(True)
                prediction = model(x)
                loss = criterion(prediction, target)

                dx = grad(loss, x)[0]
                dx = dx.view(len(target),-1)
                dxn = dx.norm(dim=-1,keepdim=True)
                if args.norm=='Linf':
                    dx = dx.sign()/np.sqrt(dx.shape[1])
                    dx = dx.view(x.shape)
                    x = x + args.AT_dt*dx.detach()
                    v = x-x0
                    v.clamp_(-args.AT_ball,args.AT_ball)
                    x = x0 + v
                    x = x.detach()
                else:
                    raise NotImplementedError('Other norms besides Linf not implemented')

            for p in model.parameters():
                p.requires_grad_(True)
            x.requires_grad_(False)

            prediction = model(x)
            loss = criterion(prediction, target)



            loss.backward()

            optimizer.step()

            if np.isnan(loss.data.item()):
                raise ValueError('model returned nan during training')

            t = ttot + time.perf_counter() - tepoch
            fmt = '{:.4f}'
            logger.writerow({'index':ix,
                'time': fmt.format(t),
                'loss': fmt.format(loss.item())})

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
