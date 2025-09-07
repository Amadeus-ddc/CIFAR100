from sympy import subsets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"

mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

train_tf = transforms.Compose([transforms.RandomCrop(32,padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(mean,std)])
test_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean,std)])

base_train = datasets.CIFAR100(root="./data", train=True, transform=train_tf, download=False)
base_val = datasets.CIFAR100(root="./data", train=False, transform=test_tf, download=False)

n_total = len(base_train)
n_val = 5000
n_train = n_total - n_val
g = torch.Generator().manual_seed(42)
perm = torch.randperm(n_total,generator=g).tolist()
train_idx = perm[:n_train]
val_idx = perm[n_train:]

train_set = Subset(base_train,train_idx)
val_set = Subset(base_val,val_idx)

batch_size = 128
num_workers = 2
pin = True if torch.cuda.is_available() else False

train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,
                          num_workers=num_workers,pin_memory=pin,drop_last=True)
val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False,
                        num_workers=num_workers,pin_memory=pin)
test_loader = DataLoader(base_val,batch_size=batch_size,shuffle=False,
                         num_workers=num_workers,pin_memory=pin)

optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=5e-4,nesterov=True)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(True)
        
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=out_ch))

    def forward(self, x):
        
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = self.make_layer(64, 64, 2, stride_first=1)
        self.layer2 = self.make_layer(64, 128, 2, stride_first=2)
        self.layer3 = self.make_layer(128, 256, 2, stride_first=2)
        self.layer4 = self.make_layer(256, 512, 2, stride_first=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,100)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)  
                nn.init.constant_(m.bias, 0.)  
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    bn2 = m.residual_function[4]
                    assert isinstance(bn2, nn.BatchNorm2d)  
                    bn2.weight.zero_()                      

    def make_layer(self, in_ch, out_ch, num_blocks, stride_first):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride_first))
        for _ in range(num_blocks-1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)                      
        out = torch.flatten(out, 1)              
        out = self.fc(out)                  
        
        return out














            




