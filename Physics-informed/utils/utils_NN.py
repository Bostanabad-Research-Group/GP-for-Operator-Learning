import torch
import torch.nn as nn
import numpy as np
import torch.linalg as linalg
import time
from tqdm import tqdm
import dill

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

import torch.nn.functional as F
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(5, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x , y , u_bc):
        batchsize = x.shape[0]
        x = x.view(batchsize , 30 , 30).unsqueeze(-1)
        y = y.view(batchsize , 30 , 30).unsqueeze(-1)
        u = u_bc.view(batchsize , 116 , 3)#.unsqueeze(-1).unsqueeze(-1).repeat(1,1,100,1)
        # u = reconstruct_images(u , [30 , 30 ,3])
        #print(f"u , x , y {u.shape , x.shape , y.shape}")
        x = torch.cat([u , x , y] , dim = -1)
        size_x, size_y = 30 , 30#x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.squeeze(-1).reshape(batchsize , -1)
        x = x.view(batchsize , 3 , 900)
        #print(x.shape)
        return x
    
class Model_DeepONet(nn.Module):
    def __init__(self, trunk, branch) -> None:
        super().__init__()
        self.trunknet = trunk
        #self.branchnet = branchnet
        self.branchnet = branch
        #self.productnet= productnet

    def forward(self, x, input):
        trunkout = self.trunknet(x)
        branchout = self.branchnet(input)
        out = torch.einsum('ij,ij->i', trunkout, branchout)

        #out =  trunkout*branchout

        return out.unsqueeze(-1)

class Model_DeepONet_LDC(nn.Module):
    def __init__(self, trunk ,branch) -> None:
        super().__init__()
        self.trunknet = trunk
        self.branchnet = branch

    def forward(self, x, input):
        trunkout = self.trunknet(x)
        branchout = self.branchnet(input)

        trunkout = trunkout.view(-1, 3, trunkout.shape[1] // 3)  # Assuming each output is of size 2 * output_dim
        branchout = branchout.view(-1, 3, branchout.shape[1] // 3)  # Assuming each output is of size 2 * output_dim

        out1 = torch.einsum('ij,ij->i', trunkout[:,0,:], branchout[:,0,:]).unsqueeze(-1)
        out2 = torch.einsum('ij,ij->i', trunkout[:,1,:], branchout[:,1,:]).unsqueeze(-1)
        out3 = torch.einsum('ij,ij->i', trunkout[:,2,:], branchout[:,2,:]).unsqueeze(-1)

        return torch.cat([out1, out2, out3], dim=1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 25),
            nn.Tanh(),
            nn.Linear(25, 50),
            nn.Tanh(),
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, input_dim),
            # nn.Sigmoid()  # Sigmoid activation for output layer to scale values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Network(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 2, layers = [40 , 40 , 40 , 40 ], activation = 'tanh', encoder = 'cosine') -> None:
        super(Network, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]

        self.H1 = nn.Linear(input_dim, layers[0])
        self.last= nn.Linear(layers[0], output_dim)

        l = nn.ModuleList()
        for i in range(len(layers)):
            l.append(nn.Linear(layers[i], layers[i]))
            #l.append(activation )

        self.layers = nn.Sequential(*l)

    def forward(self, input):
        #print(input.shape)

        H = nn.Tanh()(self.H1(input))

        for layer in self.layers:
            H = nn.Tanh()(layer(H))

        out = self.last(H)

        return out