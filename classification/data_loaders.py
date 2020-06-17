"""Data loading utils.
"""
import ipdb

from torch.utils.data import DataLoader
from torchvision import datasets


def load_cifar10(batch_size, tf=None, tf_test=None, shuffle=True):
    train_dataset = datasets.CIFAR10('data/cifar10', download=True, train=True, transform=tf)
    test_dataset  = datasets.CIFAR10('data/cifar10', download=True, train=False, transform=tf_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Data dependant init
    init_loader = DataLoader(datasets.CIFAR10('data/cifar10', train=True, download=True, transform=tf), batch_size=512, shuffle=True)
    return train_loader, test_loader, init_loader
