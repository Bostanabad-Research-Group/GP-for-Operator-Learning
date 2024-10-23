import torch
from gpytorch.constraints import Positive
from GP.models.GPregression import GPR
from .. import kernels
import dill
from utils.utils_NN import Network, Model_DeepONet
import numpy as np
from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from tqdm import tqdm

def unique_rows_in_order(A):
    A_np = np.array(A.cpu())
    _, unique_indices = np.unique(A_np, axis=0, return_index=True)
    unique_rows = A_np[np.sort(unique_indices)]
    return torch.tensor(unique_rows).to('cuda')  

def unique_idx_in_order(A):
        A_np = np.array(A.cpu())
        _, unique_indices = np.unique(A_np, axis=0, return_index=True)
        return np.sort(unique_indices)

class GP_BurgersDirichlet(GPR):
    """ GP model for the Burgers' equation.
    Arguments:
            - train_dataset: training dataset containing containing the inputs with their corresponding outputs.
            - kernel_phi: kernel used for phi (discretized input function). Default: 'MaternKernel'. Other choices: 'RBFKernel'.
            - kernel_y: kernel used for y (location where the output function is observed). Default: 'MaternKernel'. Other choices: 'RBFKernel'.
    """
    def __init__(
        self,
        train_dataset = [],
        kernel_phi:str = 'MaternKernel',
        kernel_y:str = 'MaternKernel',
        **tkwargs
    ) -> None:
        
        self.X_data = train_dataset[:][0]
        self.U_data = train_dataset[:][1]
        self.tkwargs = tkwargs
        self.q = 2
        self.alpha, self.beta, self.gamma = 1.0, 1.0, 1.0
        
        index_y = set(range(self.q))
        index_phi = set(range(self.q, self.X_data.shape[1]))

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

        super(GP_BurgersDirichlet,self).__init__(
            train_x=self.X_data,train_y=self.U_data,noise_indices=[],
            kernel_phi = kernel_phi,
            kernel_y = kernel_y,
            noise = 1e-6, fix_noise = True, lb_noise = 1e-8
        )

        branchnet = Network(input_dim = self.X_data.shape[1]-self.q, layers = [128,128,128], output_dim = 128)
        trunknet = Network(input_dim = self.q, layers = [128,128,128], output_dim = 128)
        setattr(self,'mean', Model_DeepONet(trunknet, branchnet)) 
        
        # Fix and initialize kernel hyperparameters as suggested in the paper
        self.kernel_y.raw_lengthscale.data = torch.full((1, len(index_y)), 3.0) 
        self.kernel_y.raw_lengthscale.requires_grad = False 
        self.kernel_phi.base_kernel.raw_lengthscale.data = torch.full((1, len(index_phi)), -2.0) 
        self.kernel_phi.base_kernel.raw_lengthscale.requires_grad = False 
        self.kernel_phi.raw_outputscale.data = torch.tensor(0.541)
        self.kernel_phi.raw_outputscale.requires_grad = False

        self.phi_unique = unique_rows_in_order(self.train_inputs[0][:,self.q:])
        self.y_unique = unique_rows_in_order(self.train_inputs[0][:,:self.q])

        self.to(**tkwargs)
        C = KroneckerProductLazyTensor(self.kernel_phi(self.phi_unique), self.kernel_y(self.y_unique)).to(torch.double)
        self.chol_decomp = C.cholesky()

    def calculate_PDE_loss(self, inputs, outputs):
        '''Calculate the loss function based on PDE residuals and MSE on BC/IC.
        '''
        xt = inputs.clone()[:,:self.q].requires_grad_(True)
        u0 = inputs.clone()[:,self.q:]
        U_col = outputs.clone()

        # Compute model's prediction
        m_col = self.mean(xt, u0)
        phi_col_unique = unique_rows_in_order(u0.clone().detach())
        y_col_unique = xt[unique_idx_in_order(inputs.clone()[:,:self.q]),:] 
        C_Xx = self.kernel_y(self.y_unique, y_col_unique)
        C_Aa = self.kernel_phi(self.phi_unique, phi_col_unique)
        g = KroneckerProductLazyTensor(C_Aa.t().evaluate(), C_Xx.t().evaluate()).to(torch.double)
        K_inv_offset = self.chol_decomp._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double) - self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:]).to(torch.double))
        eta = (m_col + g._matmul(K_inv_offset)).squeeze(-1)
        
        # Compute derivatives
        eta_x_and_eta_y = torch.autograd.grad(eta, xt, torch.ones_like(eta), True, True)[0]
        eta_x = eta_x_and_eta_y[:,0]
        eta_y = eta_x_and_eta_y[:,1]
        eta_xx_and_eta_xy = torch.autograd.grad(eta_x, xt, torch.ones_like(eta_x), True, True)[0]
        eta_xx = eta_xx_and_eta_xy[:,0]

        # Computer loss terms
        all_idx = torch.arange(xt.size(0)).to(xt.device)  # Generate indices for all collocation points
        BC_idx = torch.where((xt[:,0] == 0.0) | (xt[:,0] == 1.0))[0]
        IC_idx = torch.where(xt[:,1] == 0.0)[0]
        PDE_idx = torch.masked_select(all_idx, torch.logical_not(torch.isin(all_idx, torch.concat([BC_idx, IC_idx]))))  # Obtain remaining indices
        
        residual_pde = eta_y[PDE_idx] + eta[PDE_idx]*eta_x[PDE_idx] - 0.1 * eta_xx[PDE_idx]
        residual_BC = eta[BC_idx] - U_col[BC_idx]
        residual_IC = eta[IC_idx] - U_col[IC_idx]

        loss_PDE = torch.mean(residual_pde**2)
        loss_BC = torch.mean(residual_BC**2)
        loss_IC = torch.mean(residual_IC**2)
        xt.detach()

        return loss_PDE, loss_BC, loss_IC
    
    def calculate_data_loss(self):
        '''Calculate the loss function based on MLE depending on the mean function used.
        '''
        m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:]).to(torch.double)
        term2 = 0.5 * torch.dot(self.train_targets.to(torch.double) - m_train.squeeze(-1), self.chol_decomp._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double) - m_train).squeeze(-1))
        loss = term2  

        return loss
    
    def save(self, fld):
        '''Method to save the model.
        Arguments:
            - fld: directory where the model should be saved.
        '''
        with open(fld, 'wb') as f:
            dill.dump(self, f)

    def fit(self, train_dataloader, test_dataloader, fld, num_iter:int = 1000, **tkwargs) -> float:
        '''Optimize the parameters of the GP model by minimizing the combined loss function using Adam. 
        Arguments:
            - train_dataloader: training dataset.
            - test_dataloader: test dataset.
            - fld: directory where the model should be saved.
            - num_iter: number of epochs.
        '''
        self.to(**tkwargs)
        self.train()

        lr = 0.001
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= np.linspace(0,num_iter,4).tolist(), gamma=0.75)
        loss_min = float('inf')
        epochs_iter = tqdm(range(num_iter), desc='Epoch',position=0,leave=True)

        for j in epochs_iter: 
            total_loss = 0.0    

            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs, labels = batch
                X_col = torch.cat(torch.unbind(inputs, dim=0), dim=0).to(**tkwargs)
                U_col = torch.cat(torch.unbind(labels, dim=0), dim=0).to(**tkwargs)

                loss_PDE, loss_BC, loss_IC = self.calculate_PDE_loss(X_col, U_col)
                loss_data = self.calculate_data_loss()
                self.alpha, self.beta, self.gamma = self.compute_dynamic_weights_ic(loss_PDE, loss_BC, loss_IC, loss_data)
                loss = self.gamma*loss_data + loss_PDE + self.alpha*loss_BC + self.beta*loss_IC
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            if j % 50 == 0:
                rl2error = self.evaluate_error(test_dataloader)
                
            desc = f'Epoch {j} - loss {loss.item():.3e} - relative l2 error {rl2error * 100:.2f}%'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)

            average_loss = total_loss / len(train_dataloader)
            
            if average_loss < loss_min:
                self.save(fld = fld)
                loss_min = average_loss
    
    def evaluate_error(self, test_dataloader):
        '''Evaluate model on test data.
        Arguments:
            - test_dataloader: test dataset.

        Returns:
            - Test relative L2 error over 200 test samples.
        '''
        error = []
        m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])
        C_inv_offset = self.chol_decomp._cholesky_solve(self.train_targets.unsqueeze(-1).to(torch.double) - m_train.to(torch.double))

        for batch in test_dataloader:
            inputs, labels = batch
            X_test = torch.cat(torch.unbind(inputs, dim=0), dim=0)
            U_test = torch.cat(torch.unbind(labels, dim=0), dim=0)
            
            phi_col_unique = unique_rows_in_order(X_test[:,self.q:])
            y_col_unique = unique_rows_in_order(X_test[:,:self.q])

            C_yY = self.kernel_y(y_col_unique, self.y_unique).to(torch.double)
            C_Phiphi = self.kernel_phi(self.phi_unique, phi_col_unique).to(torch.double)
            
            m_col = self.mean(X_test[:,:self.q], X_test[:,self.q:])
                
            C_inv_offset_mat = C_inv_offset.reshape(C_Phiphi.shape[0], -1).t()
            U_pred = m_col.squeeze(-1) + (C_yY @ C_inv_offset_mat @ C_Phiphi).squeeze(-1)

            e = np.mean(np.linalg.norm(U_pred.squeeze().cpu().detach().numpy()-U_test.cpu().squeeze().detach().numpy(), axis = -1) / np.linalg.norm(U_test.cpu().squeeze().detach().numpy(), axis = -1)) 
            error.append(e)

            if len(error) == 200:
                break

        return np.mean(error)
    
    def compute_dynamic_weights_ic(self, pde_loss, bc_loss, ic_loss, data_loss):
        lambdaa = 0.1
        params_to_update = [param for param in self.parameters() if param.requires_grad]
        
        delta_pde_teta = torch.autograd.grad(pde_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_pde_teta if p is not None]
        delta_pde_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))
        
        delta_bc_teta = torch.autograd.grad(self.alpha * bc_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_bc_teta if p is not None]
        delta_bc_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

        delta_IC_teta = torch.autograd.grad(self.beta * ic_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values2 = [p.reshape(-1,).cpu().tolist() for p in delta_IC_teta if p is not None]
        delta_IC_teta_abs = torch.abs(torch.tensor([v for val in values2 for v in val]))

        delta_data_teta = torch.autograd.grad(self.gamma *data_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values3 = [p.reshape(-1,).cpu().tolist() for p in delta_data_teta if p is not None]
        delta_data_teta_abs = torch.abs(torch.tensor([v for val in values3 for v in val]))

        temp2 = torch.max(delta_pde_teta_abs) / torch.mean(delta_IC_teta_abs)
        temp3 = torch.max(delta_pde_teta_abs) / torch.mean(delta_data_teta_abs)
        temp = torch.max(delta_pde_teta_abs) / torch.mean(delta_bc_teta_abs)

        return (1.0 - lambdaa) * self.alpha + lambdaa * temp, (1.0 - lambdaa) * self.beta + lambdaa * temp2, (1.0 - lambdaa) * self.gamma + lambdaa * temp3