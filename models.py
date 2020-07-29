import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def get_model(args, input_size, num_classes):
    """function to get the neural network configuration
    """
    if args.model=='alexnet':
        clf = AlexNet(input_size, num_classes, int(64*args.scale))
    elif args.model=='fc':
        clf = NeuralNetFC(input_size, args.scale, num_classes, args.depth) 
    elif args.model=='CNN4':
        clf = CNN4(input_size, int(args.scale), num_classes) 
    elif args.model=='VGG11' or args.model=='VGG13' or args.model=='VGG16' or args.model=='VGG19':
        clf = VGG(args.model, input_size, num_classes, args.scale)
    elif args.model=='resnet18':
        clf = ResNet18(input_size, num_classes, args.scale)
    elif args.model=='resnet34':
        clf = ResNet34(input_size, num_classes, args.scale)
    elif args.model=='resnet50':
        clf = ResNet50(input_size, num_classes, args.scale)
    elif args.model=='resnet101':
        clf = ResNet101(input_size, num_classes, args.scale)
    elif args.model=='resnet152':
        clf = ResNet152(input_size, num_classes, args.scale)
    if args.init=='SN':
        clf.apply(init_weights_SN) 
    elif args.init=='HN':
        clf.apply(init_weights_HN)
    
    return clf

def init_weights_SN(m):
    """ parameter initialization according to the standard normal distribution
    to use: model.apply(init_weights_SN)
    """
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0,1)
        if m.bias is not None:
            m.bias.data.normal_(0,1)
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0,1)
        if m.bias is not None:
            m.bias.data.normal_(0,1)

def init_weights_HN(m):
    """the kaiming parameter initialization with normal distribution
    to use: model.apply(init_weights_HN)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
            
class NeuralNetFC(nn.Module):
    """Custom module for a simple fully connected neural network classifier
       with custom depth and width
    """
    def __init__(self, input_size, hidden_units, num_classes, depth):
        super(NeuralNetFC, self).__init__()
        self.depth = depth
        self.input_size = input_size 
        self.hidden_units = int(hidden_units)
        self.num_classes = num_classes
        self.features = self._make_layers()
        
    def forward(self, x):
        
        x = x.view(-1, self.input_size)
        
        out = self.features(x)
        
        return out
    def _make_layers(self):
        layers = []
        if self.depth == 1:
            layers += [nn.Linear(self.input_size, self.num_classes)]
        else:
            layers += [nn.Linear(self.input_size, self.hidden_units),nn.ReLU(inplace=True)]
            for i in range(self.depth-1):
                layers += [nn.Linear(self.hidden_units, self.hidden_units),nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.hidden_units, self.num_classes)]
        return nn.Sequential(*layers) 

class CNN4(nn.Module):
    """Custom module for a simple convolutional neural network classifier"""
    def __init__(self, input_size, hidden_units, num_classes):
        super(CNN4, self).__init__()
        
        # for hidden units from 100 to 500 we 5 to 25 channels accordingly
        self.num_channels = int((hidden_units-100)/20 + 5)
        
        
        self.input_size = input_size 
        if self.input_size == 3*32*32: # for cifar10/100 type of input images
            self.fc1 = nn.Linear(24*24*self.num_channels, hidden_units)
            self.in_channels = 3
        elif self.input_size == 28*28: # for mnist type of input images
            self.fc1 = nn.Linear(20*20*self.num_channels, hidden_units)
            self.in_channels = 1
            
        
        self.conv1 = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, stride=1)
        
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1)
        
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1)
        
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=1)
        
        
            
        self.fc2 = nn.Linear(hidden_units, num_classes)
        
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        if self.input_size == 3*32*32:
            x = x.view(-1, 24*24*self.num_channels)
        elif self.input_size == 28*28: 
            x = x.view(-1, 20*20*self.num_channels)
            
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        
        # transform to logits
        return x
    
class AlexNet(nn.Module):
    """AlexNet configuration to be used for MNIST, Cifar10/100 size images
    """
    def __init__(self, input_size, num_classes, hidden_units=64):
        super(AlexNet, self).__init__()
        
        self.input_size = input_size
        
        if self.input_size == 3*32*32: # cifar10/cifar100
            self.num_in_channels = 3
            self.num_out_channels = 4
        elif self.input_size == 28*28: # mnist
            self.num_in_channels = 1
            self.num_out_channels = 1
        
        self.hidden_units = hidden_units
        self.features = nn.Sequential(
            nn.Conv2d(self.num_in_channels,hidden_units , kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_units, 3*hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden_units*3, hidden_units*6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units*6, hidden_units*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_units*4, hidden_units*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*hidden_units *self.num_out_channels , 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4*self.hidden_units*self.num_out_channels)
        x = self.classifier(x)
        return x
    
    
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """VGG configurations generalized for MNIST, CIFAR-10/100 input images with custom scale 
       that scales both the number of hidden units and channels
       the default configurations are with hidden_scale=1
    """
    def __init__(self, vgg_name, input_size, num_classes, hidden_scale=1):
        super(VGG, self).__init__()
        if input_size == 32*32*3:
            self.in_channels = 3
        elif input_size == 28*28:
            self.in_channels = 1
            self.input = 'mnist'
        
        self.scale = hidden_scale
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(int(512*hidden_scale), num_classes) # int rounds to lower integer
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        cnt = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif cnt < 3 and self.input == 'mnist':
                layers += [nn.Conv2d(in_channels, int(self.scale*x), kernel_size=3, padding=2),
                           nn.BatchNorm2d(int(self.scale*x)),
                           nn.ReLU(inplace=True)]  # for mnist changed padding from 1 to 2 for three layers and keep pading=1 for the rest
                in_channels = int(self.scale*x)
                cnt = cnt+1
            else:
                layers += [nn.Conv2d(in_channels, int(self.scale*x), kernel_size=3, padding=1),
                           nn.BatchNorm2d(int(self.scale*x)),
                           nn.ReLU(inplace=True)]
                in_channels = int(self.scale*x)
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    """ResNet configurations generalized for MNIST, CIFAR-10/100 images with custom scale
    """
    def __init__(self, block, num_blocks, input_size, num_classes, scale=1):
        super(ResNet, self).__init__()
        self.in_planes = int(scale*64)
        if input_size == 32*32*3:
            in_channels = 3
            pad = 1
        elif input_size == 28*28:
            in_channels = 1
            pad = 1 # for mnist changed padding from 1 to 4

        self.conv1 = nn.Conv2d(in_channels, int(64*scale), kernel_size=3, stride=1, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64*scale))
        self.layer1 = self._make_layer(block, int(64*scale), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*scale), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*scale), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*scale), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*scale)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(input_size, num_classes, scale):
    return ResNet(BasicBlock, [2,2,2,2], input_size, num_classes, scale)

def ResNet34(input_size, num_classes, scale):
    return ResNet(BasicBlock, [3,4,6,3], input_size, num_classes, scale)

def ResNet50(input_size, num_classes, scale):
    return ResNet(Bottleneck, [3,4,6,3], input_size, num_classes, scale)

def ResNet101(input_size, num_classes, scale):
    return ResNet(Bottleneck, [3,4,23,3], input_size, num_classes, scale)

def ResNet152(input_size, num_classes, scale):
    return ResNet(Bottleneck, [3,8,36,3], input_size, num_classes, scale)

    