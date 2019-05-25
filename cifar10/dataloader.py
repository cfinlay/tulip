import os
import warnings
import numpy as np
import torch as th

import torchnet as tnt
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.utils.data.sampler as sampler
from math import floor

def cutout(mask_size,channels=3):
    if channels>1:
        mask_color=tuple([0]*channels)
    else:
        mask_color=0

    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    if mask_size >0:
        def _cutout(image):
            image = np.asarray(image).copy()

            if channels >1:
                h, w = image.shape[:2]
            else:
                h, w = image.shape

            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            square = image[ymin:ymax, xmin:xmax]
            avg = np.mean(square, axis=(0,1))
            image[ymin:ymax, xmin:xmax] = avg
            if channels==1:
                image = image[:,:,None]
            return image
    else:
        def _cutout(image):
            return image

    return _cutout



def cifar10(datadir, training_transforms=[], mode='train', transform=True, subset=None, **kwargs):

    assert mode in ['train', 'test', 'val']

    train=mode=='train'

    root = os.path.join(datadir,'cifar10')

    if mode in ['train','test']:
        if train and transform:
            tlist = [transforms.RandomCrop(32,padding=4, fill=128),
                     transforms.RandomHorizontalFlip(),
                    *training_transforms,
                     transforms.ToTensor()]
        else:
            tlist = [transforms.ToTensor()]

        transform = transforms.Compose(tlist)

        ds = CIFAR10(root, download=True, train=train, transform=transform)
        sample_size = 50000 if train else 10000
    else:
        filename = 'cifar10.1'

        label_filename = 'cifar10.1_v6_labels.npy'
        imagedata_filename = 'cifar10.1_v6_data.npy'

        label_filepath = os.path.join(root, label_filename)
        imagedata_filepath = os.path.join(root, imagedata_filename)

        try:
            labels = np.load(label_filepath)
            data = np.load(imagedata_filepath)
        except FileNotFoundError as e:
            raise type(e)('Download CIFAR10.1 .npy files from https://github.com/modestyachts/CIFAR-10.1 '
                  'and place in %s'%root)

        ds = tnt.dataset.TensorDataset([data, labels])
        augment = transforms.ToTensor()
        ltrans  = lambda x: np.array(x, dtype=np.int_)
        ds = ds.transform({0:augment, 1:ltrans})
        sample_size=2000


    # check if we're looking at a range
    if isinstance(subset, range):
        indices = np.arange(subset.start, subset.stop)
    elif isinstance(subset, tuple) and len(subset)==2:
        indices = np.arange(subset[0], subset[1])
    elif isinstance(subset, np.ndarray):
        indices = subset
    elif isinstance(subset, float):
        if (subset > 0. and subset < 1.):
            num_samples = floor(subset  * sample_size)
            assert num_samples >0
            indices = np.random.choice(sample_size, num_samples)
        else:
            raise ValueError('subset fraction must be between 0 and 1')
    elif subset is not None:
        raise ValueError('Invalid subset parameter.')

    if subset:

        # according to Pytorch docs shuffle cannot be true if we are using a sampler
        # so we're going to turn it off in case that it's on
        kwargs['shuffle'] = False

        dataloader = th.utils.data.DataLoader(ds,
                sampler=sampler.SubsetRandomSampler(indices), **kwargs)
        dataloader.Nsamples = indices.size

    else:
        dataloader = th.utils.data.DataLoader(ds, **kwargs)
        if mode=='train':
            dataloader.Nsamples = 50000
        elif mode=='test':
            dataloader.Nsamples = 10000
        else:
            dataloader.Nsamples = 2000

    dataloader.classes = 10
    dataloader.image_shape = (3, 32, 32)


    return dataloader
