import torch
import torch.nn as nn
import torch.nn.functional as F

# ********************************************************************
#   D E C O N V O L U T I O N   AND   R E S I D U A L   M O D E L
# ********************************************************************
class ResblocksDeconv(nn.Module):
    def __init__(self, in_channels, outsize):
        super(ResblocksDeconv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(128)

        self.residualBlock_1 = ResidualBlock(128, 128)
        self.residualBlock_2 = ResidualBlock(128, 128)
        self.residualBlock_3 = ResidualBlock(128, 128)
        self.residualBlock_4 = ResidualBlock(128, 128)
        self.residualBlock_5 = ResidualBlock(128, 128)
        
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias = False)
        self.bn6 = nn.BatchNorm2d(32)
        
        self.deconv7 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(32)
    
        self.deconv8 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride = 1, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(3)
    
        self.in_channels = in_channels
        self.outsize = outsize

    def __call__(self, x):        
        h = self.conv1(x)
        h = self.maxp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        
        h = self.conv2(h)
        h = self.maxp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        
        h = self.conv3(h)
        h = self.maxp3(h)
        h = self.bn3(h)
        h = F.relu(h)

        h = self.residualBlock_1(h)
        h = self.residualBlock_2(h)
        h = self.residualBlock_3(h)
        h = self.residualBlock_4(h)
        h = self.residualBlock_5(h)
        
        h = self.deconv4(h)
        h = self.bn4(h)
        h = F.relu(h)
                
        h = self.deconv5(h)
        h = self.bn5(h)
        h = F.relu(h)

        h = self.deconv6(h)
        h = self.bn6(h)
        h = F.relu(h)
        
        h = self.deconv7(h)
        h = self.bn7(h)
        h = F.relu(h)
        
        h = self.deconv8(h)
        h = self.bn8(h)
        y = (torch.tanh(h) + 1) / 2
        return y
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn1= nn.BatchNorm2d(out_channels)
        self.bn2= nn.BatchNorm2d(out_channels)
        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, x, finetune = False):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        y = F.relu(h + x)

        return y

