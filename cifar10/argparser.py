"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Adversarial robustness through gradient regularization terms')
parser.add_argument('datadir', type=str, metavar='DIR',
        help='storage directory where datasets are kept')

optional = parser.add_argument_group('Optional arguments')
optional.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status (default: 100)')
optional.add_argument('--logdir', type=str, default=None,metavar='DIR',
        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')
optional.add_argument('--seed', type=int, default=None, metavar='S',
        help='random seed (default: int(time.time()) )')
optional.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
optional.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet34',
        help='Model architecture (default: ResNet34)')

group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
group0.add_argument('--cutout',type=int, default=16, metavar='N',
        help = 'Cutout size (default: 16)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]', help='List of epochs and multiplier '
        'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.008]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
       help='SGD momentum parameter (default: 0.9)')


group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--decay',type=float, default=5e-4, metavar='L',
        help='Lagrange multiplier for weight decay (sum '
        'parameters squared) (default: 5e-4)')
group2.add_argument('--norm', type=str, choices=['L1','L2','Linf'],default='L2',
        help='norm for gradient penalty, wrt model inputs. (default: L2)'
        ' Note that this should be dual to the norm measuring adversarial perturbations')
group2.add_argument('--penalty', type=float, default=0, metavar='L',
        help='Lagrange multiplier for regularizer (squared norm gradient wrt input)')
group2.add_argument('--h', type=float, default=1e-2, metavar='H',
        help='finite difference step size (default: 1e-2)')
group2.add_argument('--fd-order', type=str, choices=['O1','O2'], default='O1',
        help='accuracy of finite differences (default: O1)')
