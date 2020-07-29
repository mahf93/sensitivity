import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import math
from torch.autograd import grad
from datasets import *
from models import *
import simplejson
import argparse
import pickle

class Results(object):
    """a custom object used to save the results of an experiment
    """
    def __init__(self, v_loss_avg, v_loss_std, t_loss_avg, t_loss_std, v_acc_avg, v_acc_std, t_acc_avg, t_acc_std, sen_avg, sen_std):
        self.vlossavg = v_loss_avg
        self.vlossstd = v_loss_std
        self.tlossavg = t_loss_avg
        self.tlossstd = t_loss_std
        
        self.vaccavg = v_acc_avg
        self.vaccstd = v_acc_std
        self.taccavg = t_acc_avg
        self.taccstd = t_acc_std
        
        self.senavg = sen_avg
        self.senstd = sen_std

def train(args, opt, epoch, my_train_loader, clf, device):
    """Function for training the network
       In each epoch it trains over each training sample once
       over the mini batches and returns the average loss and accuracy
    """
    
    # set model in training mode (need this because of dropout)
    clf.train() 
    
    train_loss = 0
    correct = 0
    
    criteria = torch.nn.CrossEntropyLoss()
    
    # dataset API gives us pythonic batching 
    for batch_id, (data, label) in enumerate(my_train_loader):
        data = Variable(data).to(device)
        target = Variable(label).to(device)
        
        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        outs = clf(data)
        loss = criteria(outs, target)
        
        # taking the average over training losses
        train_loss += loss.item()
        
        # finding the accuracy
        _, pred = torch.max(outs.data, 1)
        correct += pred.eq(target.data).sum().item()
        
        loss.backward()
        opt.step()
        
    train_loss /= len(my_train_loader)
    accuracy = 100. * correct / len(my_train_loader.dataset)
    
    return train_loss, accuracy

def test(args, test_loader, clf, device):
    """function for evaluating the model on the unseen data
    """
    with torch.no_grad():
        clf.eval()
        criteria = nn.CrossEntropyLoss()
        
        test_loss = 0
        correct = 0
        
        for data, target in test_loader:
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            
            
            output = clf(data)
            
            
            # find loss
            test_loss += criteria(output, target).item()
            
            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            correct += pred.eq(target.data).sum().item()
            
            
            
        test_loss /= len(test_loader) # loss function already averages over batch sizes
        accuracy = 100. * correct / len(test_loader.dataset)
        
        
        return test_loss, accuracy
    
def find_sen(args, test_loader, clf, device, num_classes):
    """function for computing sensitivity on the unseen data
    """
    with torch.no_grad():
        clf.eval()
        
        errors = []
        outs = []
        for data, target in test_loader:
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            
            noise = Variable(data.new(data.size()).normal_(0,args.sigma)).to(device)
            
            data_noisy = data + noise
            
            output = clf(data)
            output_noisy = clf(data_noisy)
            
            error = output_noisy - output
            error = error.cpu().detach().numpy() # shape = batch_size*num_classes
            
            output = output.cpu().detach().numpy()
            
            # going through batch:
            for ind in np.arange(np.shape(target.data.tolist())[0]):
                temp = 0
                # finding the "unspecific" sensitivity
                for i in range(num_classes):
                    temp += (error[ind][i])
                errors.append(temp/num_classes)

                
        errors = np.asarray(errors)
        return np.var(errors)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Sensitivity Code")
    parser.add_argument("--model", type=str, choices=['alexnet', 'fc', 'CNN4', 'VGG11','VGG13','VGG16','VGG19','resnet18','resnet34', 'renset50', 'resnet101', 'resnet152'], help="the neural network configuration")
    parser.add_argument("--init", type=str, choices=['SN', 'HN'], default= None, help="the parameter initialization")
    parser.add_argument("--scale", type=float, default=1, help="the scale to use for the configuration, for FC and CNN it is the number of hidden units")  
    parser.add_argument("--depth", type=int, default=4, help="the number of hidden layers of FC") 
    parser.add_argument("--dataset", type=str, default="mnist", choices=['mnist', 'cifar10', 'cifar100'], help="the dataset to use for training and testing")
    parser.add_argument("--numsamples", type=int, default=12800, help="number of training samples to train on")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size of both the training and the testing sets")
    parser.add_argument("--numepochs", type=int, default=2000, help="maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="adam learning rate")
    parser.add_argument("--sigma", type=float, default=0.1, help="the standard deviation of the injected random input noise")
    parser.add_argument("--numavgs", type=int, default=10, help="number of runs to take avg over")
    parser.add_argument("--lossthreshold", type=float, default=0.00001, help="the training loss threshold to stop the optimization")
    parser.add_argument("--logevery", type=int, default=1000, help="frequency of printing the training loss and accuracy")
    parser.add_argument("--filename", type=str, default='temp.pkl', help="filename to save the results to")
    args = parser.parse_args()
    
    # data set parameters
    if args.dataset == 'mnist':
        input_size = 28*28
        num_classes = 10
    elif args.dataset == 'cifar10':
        input_size = 32*32*3
        num_classes = 10
    elif args.dataset == 'cifar100':
        input_size = 32*32*3
        num_classes = 100
        
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data loader
    my_train_loader, my_test_loader = get_data_loader(args.dataset, batch_size=args.batchsize, num_samples=args.numsamples)
    
    vLosses = []
    tLosses = []
    vACCs = []
    tACCs = []
    Sen = []
    avg_epochs = 0
    
    for cnt in range(args.numavgs):
    
        clf = get_model(args, input_size, num_classes).to(device)
        #clf = nn.DataParallel(clf) # to run on multiple gpus
        
        
        opt = optim.Adam(clf.parameters(), lr=args.lr)
        
        epoch = 0
        inc = 0
        # train the network
        while epoch < args.numepochs:

            # train the network
            t_loss, t_acc = train(args, opt, epoch, my_train_loader, clf, device)
            
            if epoch%args.logevery == 0:
                print('epoch: ', epoch, ' loss: ', t_loss, ' acc: ', t_acc)
            
            if t_loss < args.lossthreshold:
                # we want to go below threshold for 10 times
                inc += 1
            if inc == 10 or epoch==args.numepochs-1:
                if cnt == 5:
                    print('Training is done after %d epochs' % epoch, 'loss: ', t_loss, 'acc: ', t_acc)
                avg_epochs += epoch

                # to come out of the inner loop
                epoch = args.numepochs -1 
            epoch += 1
            
        sen = find_sen(args, my_test_loader, clf, device, num_classes)
        
        ce, acc = test(args, my_test_loader, clf, device)
        if cnt == 5:
            print('val loss: ', ce, ' acc: ', acc)
        
        vLosses.append(ce)
        tLosses.append(t_loss)
        vACCs.append(acc)
        tACCs.append(t_acc)
        Sen.append(sen)
    
    avg_epochs /= args.numavgs
    
    avg_val_loss = np.mean(vLosses)
    std_val_loss = np.std(vLosses)
    
    avg_train_loss = np.mean(tLosses)
    std_train_loss = np.std(tLosses)
    
    avg_val_acc = np.mean(vACCs)
    std_val_acc = np.std(vACCs)
    
    avg_train_acc = np.mean(tACCs)
    std_train_acc = np.std(tACCs)
    
    avg_sen = np.mean(Sen)
    std_sen = np.std(Sen)
    
    # save the results
    with open(args.filename, 'wb') as output:
        rst1 = Results(avg_val_loss, std_val_loss, avg_train_loss, std_train_loss, avg_val_acc, std_val_acc, avg_train_acc, std_train_acc, avg_sen, std_sen)
        pickle.dump(rst1, output, pickle.HIGHEST_PROTOCOL)
        
    
if __name__ == "__main__":
    main() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    