from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets

n_bits = 8


# def preprocess(x):
#     # Follows:
#     # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

#     x = x * 255  # undo ToTensor scaling to [0,1]

#     n_bins = 2**n_bits
#     if n_bits < 8:
#       x = torch.floor(x / 2 ** (8 - n_bits))
#     x = x / n_bins

#     return x


# def postprocess(x):
#     x = torch.clamp(x, 0, 1)
#     x = x * 2**n_bits
#     return torch.clamp(x, 0, 255).byte()

def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**n_bits
    if n_bits < 8:
      x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x

cifar10_normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
cifar10_unnormalize = transforms.Normalize(mean=[-x / 255.0 for x in [125.3, 123.0, 113.9]], std=[1/(x / 255.0) for x in [63.0, 62.1, 66.7]])
def postprocess(x, dataset):
    if dataset == 'cifar10':
      pass
    if dataset == 'mnist':
      pass

    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**n_bits
    return torch.clamp(x, 0, 255).byte()

def get_CIFAR100(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 100

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'CIFAR100'
    train_dataset = datasets.CIFAR100(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=True)

    test_dataset = datasets.CIFAR100(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=True)

    return image_shape, num_classes, train_dataset, test_dataset

def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'CIFAR10'
    train_dataset = datasets.CIFAR10(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=True)

    test_dataset = datasets.CIFAR10(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=True)

    return image_shape, num_classes, train_dataset, test_dataset


def get_MNIST(augment, dataroot, download):
    assert not augment
    image_shape = (32, 32, 1)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.Resize(32), transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'MNIST'
    train_dataset = datasets.MNIST(path, train=True,
                                  transform=transform,
                                  target_transform=one_hot_encode,
                                  download=True)

    test_dataset = datasets.MNIST(path, train=False,
                                 transform=transform,
                                 target_transform=one_hot_encode,
                                 download=True)

    return image_shape, num_classes, train_dataset, test_dataset

def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'SVHN'
    train_dataset = datasets.SVHN(path, split='train',
                                  transform=transform,
                                  target_transform=one_hot_encode,
                                  download=True)

    test_dataset = datasets.SVHN(path, split='test',
                                 transform=transform,
                                 target_transform=one_hot_encode,
                                 download=True)

    return image_shape, num_classes, train_dataset, test_dataset
