import torch.nn.functional as F 
import torch.nn as nn

#-------------------------------------------------------------------------------
# Define Helper function for Convolutional Layers
def conv_layers(in_channels, out_channels, kernel_size, stride=2,
                padding=1, batch_norm=True):
    layers = []
    convolutional_layers = nn.Conv2d(in_channels,out_channels, kernel_size,
                stride, padding, bias=False)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers) 
#-------------------------------------------------------------------------------
# Define Helper function for Transpose Convolutional Layers
def tconv_layers(in_channels, out_chanels, kernel_size, stride=2,
                padding=1, batch_norm=True):
    layers = []
    transpose_convolutional_layers = nn.ConvTranspose2d(in_channels, out_chanels, 
                kernel_size, stride, padding, bias=False)
    layers.append(transpose_convolutional_layers)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_chanels))

    return nn.Sequential(*layers) 
#-------------------------------------------------------------------------------
# Define a Discirminator Class, the Discriminator | INCOMPLETE
class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # h,w = 256x256 d = 3
        self.conv_l1 = conv_layers(3, conv_dim, kernel_size=4, batch_norm=False)
        # h,w = 128x128 d = 64
        self.conv_l2 = conv_layers(conv_dim, conv_dim*2, kernel_size=4)
        # h,w = 64x64 d = 128
        self.conv_l3 = conv_layers(conv_dim*2, conv_dim*4, kernel_size=4)
        # h,w = 32x32 d = 256
        self.conv_l4 = conv_layers(conv_dim*4, conv_dim*8, kernel_size=4)
        # h,w = 16x16 d = 512

        self.output = nn.Linear(conv_dim*8*16*16,1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        x = self.conv_l1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_l2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_l3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_l4(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = x.view(-1, self.conv_dim*8*16*16)
        x = self.output(x)

        return x

#-------------------------------------------------------------------------------
# Define a Generator Class, the Generator | INCOMPLETE
class Generator(nn.Module):
    def __init_(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.input = nn.Linear(z_size, conv_dim*8*16*16)

        self.tconv_l1 = tconv_layers(conv_dim*8, conv_dim*4, kernel_size=4)
        self.tconv_l2 = tconv_layers(conv_dim*4, conv_dim*2, kernel_size=4)
        self.tconv_l3 = tconv_layers(conv_dim*2, conv_dim, kernel_size=4)
        self.tconv_l4 = tconv_layers(conv_dim, 3, kernel_size=4, batch_norm=False)
    
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 256x256x3 Tensor image as output
        """
        x = self.input(x)
        x = x.view(-1, self.conv_dim*8, 16, 16) # Check for right dimension

        x = self.tconv_l1(x)
        x = F.relu(x)
        x = self.tconv_l2(x)
        x = F.relu(x)
        x = self.tconv_l3(x)
        x = F.relu(x)
        x = self.tconv_l4

        x = F.tanh(x)

        return x
#-------------------------------------------------------------------------------