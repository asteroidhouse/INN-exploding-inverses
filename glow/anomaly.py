import os
import ipdb
import pickle

import numpy as np
import sklearn.metrics as sk

import torch
import torchvision
import torchvision.transforms as transforms

DATA_DIR='./data'
recall_level_default = 0.9



def _load_ood_dataset(ood_dataset_name, opt_config):
    if ood_dataset_name == 'notMNIST':
        # N_ANOM = 2000
        pickle_file = os.path.join(DATA_DIR, 'notMNIST.pickle')
        with open(pickle_file, 'rb') as f:

            try:
                save = pickle.load(f, encoding='latin1')
            except TypeError:
                save = pickle.load(f)

            ood_data = save['train_dataset'][:,None] * opt_config['ood_scale']  # (20000, 1, 28, 28)
            del save
        return ood_data
    elif ood_dataset_name == 'cifar10bw':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28,28)),  # Resized to 28x28 to match the size of Omniglot digits
            transforms.ToTensor(),
        ])

        cifar10_batch_size = 10
        cifar10_testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
        cifar10_testloader = torch.utils.data.dataloader.DataLoader(cifar10_testset, batch_size=cifar10_batch_size, shuffle=False)
        cifar10_testiter = iter(cifar10_testloader)

        ood_data_list = []

        while True:
            try:
                cifar10_images, _ = cifar10_testiter.next()
                ood_data_list.append(cifar10_images)
            except StopIteration:
                break

        ood_data = torch.cat(ood_data_list, 0)
        return ood_data.numpy() * opt_config['ood_scale']  # For consistency, all parts of this function return numpy arrays  (10000, 1, 28, 28)
    elif ood_dataset_name == 'gaussian':
        return np.clip(.5 + np.random.normal(size=(opt_config['n_anom'], 3, 28, 28)), a_min=0, a_max=1)
    elif ood_dataset_name == 'uniform':
        return np.random.uniform(size=(opt_config['n_anom'], 1, 28, 28)) * opt_config['ood_scale']
    elif ood_dataset_name == 'rademacher':
        return (np.random.binomial(1, .5, size=(opt_config['n_anom'], 3, 32, 32)))
    elif ood_dataset_name == 'texture3':
        return torch.load(os.path.join(DATA_DIR, 'dtd.t7')).numpy() / 255.
    elif ood_dataset_name == 'places3':
        return torch.load(os.path.join(DATA_DIR, 'places.t7')).numpy() / 255.
    elif ood_dataset_name == 'svhn':
        ds = torchvision.datasets.SVHN('.', split='test', transform=None, target_transform=None, download=True)
        data = ds.data
        np.random.shuffle(data)
        return data[:10000] / 255.
    # LSUN, iSUN, and TinyImageNet are based on ODIN: https://github.com/facebookresearch/odin/blob/master/code/cal.py
    elif ood_dataset_name == 'lsun':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'LSUN'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'lsun_resized':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'LSUN_resize'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'isun':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'iSUN'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'tinyimagenet':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'Imagenet'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'tinyimagenet_resized':
        transform = transforms.Compose([transforms.ToTensor()])
        ood_data = torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, 'Imagenet_resize'), transform=transform)
        return np.stack([img.numpy() for (img, label) in ood_data])
    elif ood_dataset_name == 'cifar-fs-train-test':
        cifar = torchvision.datasets.CIFAR100(DATA_DIR, train=False, transform=None)
        return np.transpose(cifar.data[np.array(cifar.targets)<64], (0,3,1,2)) / 255
    elif ood_dataset_name == 'cifar-fs-test':
        cifar = torchvision.datasets.CIFAR100(DATA_DIR, train=False, transform=None)
        return np.transpose(cifar.data[np.array(cifar.targets)>=80], (0,3,1,2)) / 255
    else:
        raise ValueError('invalid OOD type')


def load_ood_data(ooc_config):
    # Note:
    # Most of the time, the test set of our in-distribution has about 10k
    # examples.  This was a choice made while preparing the OOD datasets.
    # So, datasets like 'texture3' which requires the users to download a
    # .t7 files has only 10k examples.

    # Load OOD dataset
    ood_dataset = _load_ood_dataset(ooc_config['name'], ooc_config)[:ooc_config['n_anom']]
    ood_dataset = np.transpose(ood_dataset,(0,2,3,1)) # (B, 3, H, W) -> (B, H, W, 3)
    assert ood_dataset.max() <= 1
    ood_dataset = (ood_dataset * 255).astype('uint8')
    assert (ood_dataset.shape[3] in [1,3])
    return ood_dataset


