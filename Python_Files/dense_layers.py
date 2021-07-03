import torch
import torch.nn as nn
import torch.nn.functional as F
#Inception like MOdule with addition
# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
        

# class Inception_with_Addition(nn.Module):
#   def __init__(self, in_channel, out_channel):
#     super(Inception_with_Addition, self).__init__()
    
#     self.branch3x3_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

#     self.branch5x5_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2)

#     self.branch7x7_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, padding=3)
    
#     self.branch9x9_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=9, padding=4)
    

#   def forward(self, x):


#     branch3x3_2_ = self.branch3x3_2(x)

#     branch5x5_2_ = self.branch5x5_2(x)
    
#     branch7x7_2_  = self.branch7x7_2(x)
    
#     branch9x9_2_  = self.branch9x9_2(x)

#     outputs =  branch3x3_2_ + branch5x5_2_ + branch7x7_2_ + branch9x9_2_

#     return outputs 


class Inception_with_Addition(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Inception_with_Addition, self).__init__()
    
    self.branch3x3_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

    self.branch5x5_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2)

    self.branch7x7_2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, padding=3)
    

  def forward(self, x):


    branch3x3_2_ = self.branch3x3_2(x)

    branch5x5_2_ = self.branch5x5_2(x)
    
    branch7x7_2_  = self.branch7x7_2(x)

    outputs =  branch3x3_2_ + branch5x5_2_ + branch7x7_2_ 

    return outputs 
    
# Squeeze-and-Excitation Network, (Inception or Residual) 
class SqEx_Res(nn.Module):

    def __init__(self, n_features, reduction=4):
        super(SqEx_Res, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU() 
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        # residual = x
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1) # change [B, C, H,  W] to [1, 1, 1,  C]
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y

        # y = residual + y

        return y
# Inception Module        
class Inception_Module(nn.Module):
  def __init__(self, in_channel, pool_features=48):
    super(Inception_Module, self).__init__()
    
    self.branch1x1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=1)

    # self.branch3x3_1 = nn.Conv2d(in_channels=in_channel, out_channels=48, kernel_size=1)
    self.branch3x3_2 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3, padding=1)

    # self.branch5x5_1 = nn.Conv2d(in_channels=in_channel, out_channels=48, kernel_size=1)
    self.branch5x5_2 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=5, padding=2)

    
    # self.branch7x7_1 = nn.Conv2d(in_channels=in_channel, out_channels=48, kernel_size=1)
    self.branch7x7_2 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=7, padding=3)

    self.branch_pool_1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=2, stride=2)
    self.branch_pool_2 = nn.Conv2d(in_channels=in_channel, out_channels=pool_features, kernel_size=1)

  def forward(self, x):

    branch7x7_2_  = self.branch7x7_2(x)

    branch3x3_2_ = self.branch3x3_2(x)

    branch5x5_2_ = self.branch5x5_2(x)


    outputs = [branch7x7_2_, branch3x3_2_, branch5x5_2_]

    return torch.cat(outputs, 1)
    
    
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp_Add(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=3, stride=2, padding=0, bias=True)
        #increase the number of channels using 1x1 conv            
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                          stride=1, padding= 1, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = self.conv1x1(out)
        out1 = center_crop(out, skip.size(2), skip.size(3))
        # out = torch.cat([out, skip], 1)
        out = out1 + skip
        return out

# Squeeze-and-Excitation Network, (Inception or Residual) 
class SqEx_Res(nn.Module):

    def __init__(self, n_features, reduction=4):
        super(SqEx_Res, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU() 
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        # residual = x
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1) # change [B, C, H,  W] to [1, 1, 1,  C]
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y

        # y = residual + y

        return y
        
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)
        
        self.SE_block = SqEx_Res( n_features=out_channels )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        
        out = torch.cat([out, skip], 1)
        return out

class TransitionUp_SE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)
        
        self.SE_block = SqEx_Res( n_features=out_channels )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = self.SE_block(out)
        out = center_crop(out, skip.size(2), skip.size(3))
        
        out = torch.cat([out, skip], 1)
        return out



class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]