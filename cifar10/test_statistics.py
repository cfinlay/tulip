import argparse
import os, sys

import numpy as np
import pandas as pd
import pickle as pk
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import grad
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

sys.path.append('../')
from eigen import dominant_hessian_eigs
import models





parser = argparse.ArgumentParser('Gathers statistics of a model on the test'
        'set, and saves these statistics to a pickle file in the model directory')

parser.add_argument('datadir', type=str,
        help='Directory where CIFAR-10 dataset is saved')
parser.add_argument('--model-path', type=str, required=True,metavar='PATH',
        help='Path to saved PyTorch model')
parser.add_argument('--num-images', type=int, default=10000,metavar='N',
        help='total number of images to evaluate (default: 10000)')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')
parser.add_argument('--norm', default='L2', choices=['L2', 'Linf'],
        help='norm measuring adversarial perturbations')
parser.add_argument('--loss', type=str, default='CW',
        choices=['CW','SoftCW'],
        help='loss function. Either "CW" (modified Carlini-Wagner); or "SoftCW" (default: "CW")')

parser.add_argument('--curvature', help='compute curvature statistics. This is technically only available for "SoftCW" or "CE" losses on models with "smooth" ReLUs',
        action='store_true')

args = parser.parse_args()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

transform = transforms.Compose([transforms.ToTensor()])

root = os.path.join(args.datadir)
ds = CIFAR10(root, download=False, train=False, transform=transform)

Ix = torch.arange(args.num_images)
subset = Subset(ds, Ix)

loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)


d = torch.load(args.model_path, map_location='cpu')
model = getattr(models, d['model'])()
model.load_state_dict(d['state_dict'])

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

Nsamples = args.num_images
Nc = 10


if args.loss=='CW':
    # max_{i\neq c} p_i - p_c
    def criterion(z,y):
        p = z.softmax(dim=-1)
        ix = torch.arange(z.shape[0],device=z.device)
        pc = p.clone()
        pc[ix,y] = 0.

        return pc.max(dim=-1)[0] - p[ix,y]
elif args.loss=='SoftCW':
    # softmax_{i\neq c} p_i - p_c
    def criterion(z,y):
        p = z.softmax(dim=-1)
        ix = torch.arange(z.shape[0],device=z.device)
        pc = p.clone()
        pc[ix,y] = 0.
        pcsm = (pc*10).softmax(dim=-1)*pc

        return pcsm.sum(dim=-1) - p[ix,y]
        

Loss = torch.zeros(Nsamples).cuda()
NormGradLoss = torch.zeros(Nsamples).cuda()
if args.curvature and args.norm=='L2':
    LambdaMaxLoss = torch.zeros(Nsamples).cuda()
Top1 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
Rank = torch.zeros(Nsamples,dtype=torch.int64).cuda()
Top5 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()



sys.stdout.write('\nRunning through dataloader:\n')
Jx = torch.arange(Nc).cuda().view(1,-1)
Jx = Jx.expand(args.batch_size, Nc)
for i, (x,y) in enumerate(loader):
    sys.stdout.write('  Completed [%6.2f%%]\r'%(100*i*args.batch_size/Nsamples))
    sys.stdout.flush()

    x, y = x.cuda(), y.cuda()

    x.requires_grad_(True)

    yhat = model(x)
    p = yhat.softmax(dim=-1)

    psort , jsort = p.sort(dim=-1,descending=True)
    b = jsort==y.view(-1,1)
    rank = Jx[b]
    pmax = psort[:,0]
    logpmax = pmax.log()

    p5,ix5 = psort[:,0:5], jsort[:,0:5]
    ix1 =  jsort[:,0]
    sump5 = p5.sum(dim=-1)

    loss = criterion(yhat, y)
    g = grad(loss.sum(),x)[0]
    if args.norm=='L2':
        gn = g.view(len(y),-1).norm(dim=-1)
    elif args.norm=='Linf':
        gn = g.view(len(y),-1).norm(p=1,dim=-1)

    if args.curvature and args.norm=='L2':
        lmin, lmax = dominant_hessian_eigs(lambda z: criterion(model(z),y).sum(), x, fd=False,
                                 tol=1e-3, maxiters=50)




    top1 = ix1==y
    top5 = (ix5==y.view(args.batch_size,1)).sum(dim=-1)

    ix = torch.arange(i*args.batch_size, (i+1)*args.batch_size,device=x.device)

    Loss[ix] = loss.detach()
    Rank[ix]= rank.detach()
    Top1[ix] = top1.detach()
    Top5[ix] = top5.detach().type(torch.uint8)
    NormGradLoss[ix] = gn.detach()
    if args.curvature:
        LambdaMaxLoss[ix] = lmax.detach()
sys.stdout.write('   Completed [%6.2f%%]\r'%(100.))

df = pd.DataFrame({'loss':Loss.cpu().numpy(),
                   'top1':np.array(Top1.cpu().numpy(),dtype=np.bool),
                   'top5':np.array(Top5.cpu().numpy(), dtype=np.bool),
                   'norm_grad_loss':NormGradLoss.cpu().numpy(),
                   'rank': Rank.cpu().numpy()})

if args.curvature and args.norm=='L2':
   df['lambda_max_loss'] = LambdaMaxLoss.cpu().numpy()

print('\n\ntop1 error: %.2f%%'%(100-df['top1'].sum()/Nsamples*100))

Lmax = NormGradLoss.max()
Lmean = NormGradLoss.mean()
dualnorm = 'L1' if args.norm=='Linf' else 'L2'
print('mean & max gradient norm (%s): %.2f, %.2f'%(dualnorm, Lmean, Lmax))
if args.curvature and args.norm=='L2':
    Cmax = LambdaMaxLoss.max()
    Cmean = LambdaMaxLoss.mean()
    print('mean & max curvature (%s): %.2g, %.2g'%(dualnorm, Cmean, Cmax))

LossGap = (-Loss).clamp(0)
Lbound = LossGap/Lmax
df['Lbound'] = Lbound.cpu().numpy()
print('mean 1st order lower bound on adversarial distance (%s): %.2g'%(args.norm, Lbound.mean()))

if args.curvature and args.norm=='L2':
    Cbound = 1/Cmax*(-NormGradLoss + (NormGradLoss.pow(2) + 2*Cmax*LossGap).sqrt())
    df['Cbound'] = Cbound.cpu().numpy()
    print('mean 2nd order lower bound on adversarial distance (L2): %.2g'%Cbound.mean())



ix1 = np.array(df['top1'], dtype=bool)
ix5 = np.array(df['top5'], dtype=bool)
ix15 = np.logical_or(ix5,ix1)
ixw = np.logical_not(np.logical_or(ix1, ix5))

df['type'] = pd.DataFrame(ix1.astype(np.int8) + ix5.astype(np.int8))
d = {0:'mis-classified',1:'top5',2:'top1'}
df['type'] = df['type'].map(d)
df['type'] = df['type'].astype('category')


basename = args.model_path.split('.pth.tar')
pklpath = basename[0]+'-stats-%s.pkl'%args.norm
df.to_pickle(pklpath)
