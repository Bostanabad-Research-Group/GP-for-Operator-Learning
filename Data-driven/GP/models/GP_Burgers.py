import torch
import numpy as np
import math
import gpytorch
from torch import nn
import torch.nn.functional as F 
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior
from .GPregression import GPR
from .. import kernels
from tqdm import tqdm
import dill
from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from utils.utils_NN import Network, Model_DeepONet
from copy import deepcopy

def unique_rows_in_order(A):
    A_np = np.array(A.cpu())
    _, unique_indices = np.unique(A_np, axis=0, return_index=True)
    unique_rows = A_np[np.sort(unique_indices)]
    
    return torch.tensor(unique_rows).to('cuda')

class GP_Burgers(GPR):
    """ GP model for the Burgers' equation.
    Arguments:
            - X_data: inputs of the training data [y, \phi(u)].
            - U_data: outputs of the training data [\psi(v)].
            - kernel_phi: kernel used for phi (discretized input function). Default: 'MaternKernel'. Other choices: 'RBFKernel'.
            - kernel_y: kernel used for y (location where the output function is observed). Default: 'MaternKernel'. Other choices: 'RBFKernel'.
            - mean_type: mean function used. Default: 'zero'. Other choices: 'DeepONet' or 'FNO'.
    """
    def __init__(
        self,
        X_data:torch.Tensor,
        U_data:torch.Tensor,
        kernel_phi:str = 'MaternKernel',
        kernel_y:str = 'MaternKernel',
        mean_type = 'zero',
        **tkwargs
    ) -> None:
        
        self.tkwargs = tkwargs
        self.mean_type = mean_type
        self.q = 1

        index_y = set(range(self.q))
        index_phi = set(range(self.q, X_data.shape[1]))

        kernel_phi = getattr(kernels, kernel_phi)(
            ard_num_dims=len(index_phi),
            active_dims=torch.arange(len(index_phi)),
            lengthscale_constraint= Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2), inv_transform= lambda x: -2.0*torch.log10(x/2.0))
        )
        
        kernel_y = getattr(kernels, kernel_y)(
            ard_num_dims=len(index_y),
            active_dims=torch.arange(len(index_y)),
            lengthscale_constraint= Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
        )

        super(GP_Burgers,self).__init__(
            train_x=X_data,train_y=U_data,noise_indices=[],
            kernel_phi = kernel_phi,
            kernel_y = kernel_y,
            noise = 1e-6, fix_noise = True, lb_noise = 1e-8
        )

        if self.mean_type=='zero':
            self.mean_module = gpytorch.means.ConstantMean(prior = NormalPrior(0.,1.))
            self.mean_module.constant.data = torch.tensor([0.0])
            self.mean_module.constant.requires_grad = False

        elif self.mean_type=='DeepONet':
            branchnet = Network(input_dim = X_data.shape[1]-self.q, layers = [128,128,128], output_dim = 128)
            trunknet = Network(input_dim = self.q, layers = [128,128,128], output_dim = 128)
            setattr(self,'mean', Model_DeepONet(trunknet, branchnet)) 

        elif self.mean_type=='FNO':
            modes = 16
            width = 64
            setattr(self,'mean', FNO1d(modes, width)) 
        
        # Fix and initialize kernel hyperparameters as suggested in the paper
        if self.mean_type == 'zero':
            self.kernel_y.raw_lengthscale.data = torch.full((1, len(index_y)), 3.0)
            self.kernel_y.raw_lengthscale.requires_grad = False
            self.kernel_phi.base_kernel.raw_lengthscale.data = torch.full((1, len(index_phi)), -2.0)
            self.kernel_phi.raw_outputscale.data = torch.tensor(0.541)

        else:
            self.kernel_y.raw_lengthscale.data = torch.full((1, len(index_y)), 3.0) 
            self.kernel_y.raw_lengthscale.requires_grad = False 
            self.kernel_phi.base_kernel.raw_lengthscale.data = torch.full((1, len(index_phi)), -2.0) 
            self.kernel_phi.base_kernel.raw_lengthscale.requires_grad = False 
            self.kernel_phi.raw_outputscale.data = torch.tensor(0.541)
            self.kernel_phi.raw_outputscale.requires_grad = False

        self.phi_unique = unique_rows_in_order(self.train_inputs[0][:,self.q:])
        self.y_unique = unique_rows_in_order(self.train_inputs[0][:,:self.q])
        self.to(tkwargs['device'])

    def calculate_loss(self):
        '''Calculate the loss function based on MLE depending on the mean function used.
        '''
        C_Phi = self.kernel_phi(self.phi_unique)
        C_Y = self.kernel_y(self.y_unique)
        C = KroneckerProductLazyTensor(C_Phi, C_Y).to(torch.double)
        
        if self.mean_type == 'zero':
            term2 = 0.5 * torch.dot(self.train_targets.to(torch.double), C.cholesky()._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double)).squeeze(-1))
            term1 = 0.5 * C._logdet()
            loss =  term1 + term2  
        
        else:
            if self.mean_type == 'DeepONet':
                m_train_all = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])
            elif self.mean_type == 'FNO':
                m_train_all = self.mean(self.phi_unique.unsqueeze(-1)).squeeze(-1).flatten()        
            term2 = 0.5 * torch.dot(self.train_targets.to(torch.double) - m_train_all.to(torch.double), C.cholesky()._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double) - m_train_all.unsqueeze(-1).to(torch.double)).squeeze(-1))
            loss =  term2  

        return loss

    def fit(self, X_test, U_test, num_iter:int = 1000) -> float:
        '''Optimize the parameters of the GP model by minimizing the loss function using Adam. 
        Arguments:
            - X_test: inputs of the test data.
            - U_test: outputs of the test data.
            - num_iter: number of epochs.
        '''
        self.train()
        f_inc = math.inf
        current_state_dict = self.state_dict()

        if self.mean_type == 'zero':
            lr = 0.01
        else:
            lr = 0.001

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= np.linspace(0,num_iter,4).tolist(), gamma=0.75)
        epochs_iter = tqdm(range(num_iter), desc='Epoch', position=0,leave=True)
        
        for j in epochs_iter:        
            optimizer.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if j % 100 == 0:
                rl2error = self.evaluate_error(X_test.detach(), U_test.detach())

            desc = f'Epoch {j} - loss {loss.item():.3e} - relative l2 error {rl2error * 100:.3f}%'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)
        
        if loss.item() < f_inc:
            current_state_dict = deepcopy(self.state_dict())
            f_inc = loss.item()

        self.load_state_dict(current_state_dict)
    
    def save(self, fld):
        '''Method to save the model.
        Arguments:
            - fld: directory where the model should be saved.
        '''
        with open(fld, 'wb') as f:
            dill.dump(self, f)

    def evaluate_error(self, X_test, U_test):
        '''Evaluate model on test data.
        Arguments:
            - X_test: inputs of the test data.
            - U_test_grid: outputs of the test data.

        Returns:
            - Test relative L2 error over 200 test samples.
        '''
        error = []
        C_Phi = self.kernel_phi(self.phi_unique)
        C_Y = self.kernel_y(self.y_unique)
        C = KroneckerProductLazyTensor(C_Phi, C_Y).to(torch.double)

        if self.mean_type == 'zero':
                C_inv_offset = C.cholesky()._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double))

        else:
            if self.mean_type == 'DeepONet':
                m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])
            
            elif self.mean_type == 'FNO':
                m_train = self.mean(self.phi_unique.unsqueeze(-1)).squeeze(-1).flatten()

            C_inv_offset = C.cholesky()._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double) - m_train.unsqueeze(-1))

        for i in range(U_test.shape[0]):
            U_test_i = U_test[i,:].flatten()
            X_test_i = X_test[i,...]

            ##################################################
            phi_col_unique = unique_rows_in_order(X_test_i[:,self.q:])
            y_col_unique = unique_rows_in_order(X_test_i[:,:self.q])

            C_yY = self.kernel_y(y_col_unique, self.y_unique).to(torch.double)
            C_Phiphi = self.kernel_phi(self.phi_unique, phi_col_unique).to(torch.double)
            
            if self.mean_type == 'zero':
                C_inv_offset_mat = C_inv_offset.reshape(C_Phiphi.shape[0], -1).t()
                U_pred = (C_yY @ C_inv_offset_mat @ C_Phiphi).squeeze(-1)

            else:
                if self.mean_type == 'DeepONet':
                    m_col = self.mean(X_test_i[:,:self.q], X_test_i[:,self.q:])
                
                elif self.mean_type == 'FNO':
                    m_col = self.mean(phi_col_unique.unsqueeze(-1)).squeeze(-1).flatten()

                C_inv_offset_mat = C_inv_offset.reshape(C_Phiphi.shape[0], -1).t()
                U_pred = m_col + (C_yY @ C_inv_offset_mat @ C_Phiphi).squeeze(-1)

            e = np.mean(np.linalg.norm(U_pred.squeeze().cpu().detach().numpy()-U_test_i.cpu().squeeze().detach().numpy(), axis = -1) / np.linalg.norm(U_test_i.cpu().squeeze().detach().numpy(), axis = -1)) 
            error.append(e)

            if len(error) == 200:
                break

        return np.mean(error)

    # def evaluate_error(self, X_test, U_test):
    #     error = []
    #     C_Phi = self.kernel_phi(self.phi_unique)
    #     C_Y = self.kernel_y(self.y_unique)

    #     if self.mean_type == 'zero':
    #         residuals_matrix = (self.train_targets.reshape(C_Phi.shape[0], -1)).t().to(torch.double)
        
    #     else:
    #         if self.mean_type == 'DeepONet':
    #             m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])
        
    #         elif self.mean_type == 'FNO':
    #             m_train = self.mean(self.phi_unique.unsqueeze(-1)).squeeze(-1).flatten()

    #         residuals_matrix = (self.train_targets.reshape(C_Phi.shape[0], -1) - m_train.to(torch.double).reshape(C_Phi.shape[0], -1)).t()

    #     A = torch.inverse(C_Y.evaluate()) @ residuals_matrix @ torch.inverse(C_Phi.evaluate())

    #     for i in range(U_test.shape[0]):
    #         U_test_i = U_test[i,:].flatten()
    #         X_test_i = X_test[i,:,:]

    #         phi_col_unique = unique_rows_in_order(X_test_i[:,self.q:])
    #         y_col_unique = unique_rows_in_order(X_test_i[:,:self.q])

    #         C_Yy = self.kernel_y(y_col_unique, self.y_unique).to(torch.double)
    #         C_Phiphi = self.kernel_phi(self.phi_unique, phi_col_unique).to(torch.double)
            
    #         if self.mean_type == 'zero':
    #             U_pred = (C_Yy @ A @ C_Phiphi).squeeze(-1) 

    #         else:
    #             if self.mean_type == 'DeepONet':
    #                 m_col = self.mean(X_test_i[:,:self.q], X_test_i[:,self.q:])
                
    #             elif self.mean_type == 'FNO':
    #                 m_col = self.mean(phi_col_unique.unsqueeze(-1)).squeeze(-1).flatten()

    #             U_pred = m_col + (C_Yy @ A @ C_Phiphi).squeeze(-1)

    #         e = np.mean(np.linalg.norm(U_pred.squeeze().cpu().detach().numpy()-U_test_i.cpu().squeeze().detach().numpy(), axis = -1)/np.linalg.norm(U_test_i.cpu().squeeze().detach().numpy(), axis = -1)) 
    #         error.append(e)

    #         if len(error) == 200:
    #             break

    #     return np.mean(error)

################################################################
#  FNO. Extracted from FNO's GitHub code.
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)