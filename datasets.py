from torchvision import datasets, transforms
import torch
import numpy as np
import time


def get_data_loader(name, batch_size, num_samples=None):
    """ get test and train dataloaders
    Params
    -----
    name: the name of the dataset.
    batch_size: int
    The size of the batch.
    num_samples: int
    Default None. The number of training samples to use for training.
    """
    if name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.MNIST(root='./data', 
                                          train=False, download=True,
                                          transform=transforms.ToTensor())
    elif name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.CIFAR10(root='./data', 
                                              train=False, download=True,
                                              transform=transforms.ToTensor())
    elif name == 'cifar100':
        
        train_dataset = datasets.CIFAR100(root='./data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = datasets.CIFAR100(root='./data', 
                                          train=False, download=True,
                                          transform=transforms.ToTensor())

    if num_samples == None:
        num_samples = len(train_dataset)
     
    # in case we want to train on a random subset of the training set
    my_train_dataset, rest_train_dataset = torch.utils.data.random_split(dataset=train_dataset, lengths=[num_samples,len(train_dataset)-num_samples])

    
    # Data loader
    my_train_loader = torch.utils.data.DataLoader(dataset=my_train_dataset, num_workers = 2,
                                               batch_size=batch_size, 
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers = 2,
                                              batch_size=batch_size, 
                                              shuffle=True)
    return my_train_loader, test_loader