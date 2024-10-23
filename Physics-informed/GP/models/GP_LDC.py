import torch
from gpytorch.constraints import Positive
from GP.models.GPregression import GPR
from .. import kernels
import dill
from utils.utils_NN import Network, Model_DeepONet_LDC
import numpy as np
from gpytorch.lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor
from tqdm import tqdm
from utils.utils_data import dataloader_to_tensor

def unique_rows_in_order(A):
    A_np = np.array(A.cpu())
    _, unique_indices = np.unique(A_np, axis=0, return_index=True)
    unique_rows = A_np[np.sort(unique_indices)]
    return torch.tensor(unique_rows).to('cuda')  

def unique_idx_in_order(A):
        A_np = np.array(A.cpu())
        _, unique_indices = np.unique(A_np, axis=0, return_index=True)
        return np.sort(unique_indices)

class GP_LDC(GPR):
    """ GP model for the LDC problem.
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
        self.U_data = train_dataset[:][1][:,0]
        self.V_data = train_dataset[:][1][:,1]
        self.P_data = train_dataset[:][1][:,2]
        self.tkwargs = tkwargs
        self.q = 2
        self.alpha, self.beta = 1.0, 1.0
        
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

        super(GP_LDC,self).__init__(
            train_x=self.X_data,train_y=self.U_data,noise_indices=[],
            kernel_phi = kernel_phi,
            kernel_y = kernel_y,
            noise = 1e-6, fix_noise = True, lb_noise = 1e-8
        )

        branchnet = Network(input_dim = self.X_data.shape[1]-self.q, layers = [150,150,150], output_dim = 150)
        trunknet = Network(input_dim = self.q, layers = [150,150,150], output_dim = 150)
        setattr(self,'mean', Model_DeepONet_LDC(trunknet, branchnet)) 
        
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
        m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])
        m_col = self.mean(xt, u0)
        phi_col_unique = unique_rows_in_order(u0.clone().detach())
        y_col_unique = xt[unique_idx_in_order(inputs.clone()[:,:self.q]),:] 
    
        C_yY = self.kernel_y(y_col_unique, self.y_unique).to(torch.double)
        C_Phiphi = self.kernel_phi(self.phi_unique, phi_col_unique).to(torch.double)

        C_inv_offset_U = self.chol_decomp._cholesky_solve(self.U_data.unsqueeze(-1).to(torch.double) - m_train[:,0].unsqueeze(-1))
        C_inv_offset_V = self.chol_decomp._cholesky_solve(self.V_data.unsqueeze(-1).to(torch.double) - m_train[:,1].unsqueeze(-1))
        C_inv_offset_P = self.chol_decomp._cholesky_solve(self.P_data.unsqueeze(-1).to(torch.double) - m_train[:,2].unsqueeze(-1))
       
        C_inv_offset_mat_U = C_inv_offset_U.reshape(C_Phiphi.shape[0], -1).t()
        C_inv_offset_mat_V = C_inv_offset_V.reshape(C_Phiphi.shape[0], -1).t()
        C_inv_offset_mat_P = C_inv_offset_P.reshape(C_Phiphi.shape[0], -1).t()
        eta_u = m_col[:,0] + (C_yY @ C_inv_offset_mat_U @ C_Phiphi).squeeze(-1)
        eta_v = m_col[:,1] + (C_yY @ C_inv_offset_mat_V @ C_Phiphi).squeeze(-1)
        eta_p = m_col[:,2] + (C_yY @ C_inv_offset_mat_P @ C_Phiphi).squeeze(-1)

        # Compute derivatives
        eta_u_x_and_eta_u_y = torch.autograd.grad(eta_u, xt, torch.ones_like(eta_u), True, True)[0]
        eta_u_x = eta_u_x_and_eta_u_y[:,0]
        eta_u_y = eta_u_x_and_eta_u_y[:,1]
        eta_u_xx_and_eta_u_xy = torch.autograd.grad(eta_u_x, xt, torch.ones_like(eta_u_x), True, True)[0]
        eta_u_yx_and_eta_u_yy = torch.autograd.grad(eta_u_y, xt, torch.ones_like(eta_u_y), True, True)[0]
        eta_u_xx = eta_u_xx_and_eta_u_xy[:,0]
        eta_u_yy = eta_u_yx_and_eta_u_yy[:,1]

        eta_v_x_and_eta_v_y = torch.autograd.grad(eta_v, xt, torch.ones_like(eta_v), True, True)[0]
        eta_v_x = eta_v_x_and_eta_v_y[:,0]
        eta_v_y = eta_v_x_and_eta_v_y[:,1]
        eta_v_xx_and_eta_v_xy = torch.autograd.grad(eta_v_x, xt, torch.ones_like(eta_v_x), True, True)[0]
        eta_v_yx_and_eta_v_yy = torch.autograd.grad(eta_v_y, xt, torch.ones_like(eta_v_y), True, True)[0]
        eta_v_xx = eta_v_xx_and_eta_v_xy[:,0]
        eta_v_yy = eta_v_yx_and_eta_v_yy[:,1]

        eta_p_x_and_eta_p_y = torch.autograd.grad(eta_p, xt, torch.ones_like(eta_p), True, True)[0]
        eta_p_x = eta_p_x_and_eta_p_y[:,0]
        eta_p_y = eta_p_x_and_eta_p_y[:,1]

        # Computer loss terms
        all_idx = torch.arange(xt.size(0)).to(xt.device)  # Generate indices for all collocation points
        BC_idx = torch.where((xt[:,0] == 0.0) | (xt[:,1] == 0.0) | (xt[:,0] == 1.0) | (xt[:,1] == 1.0))[0]
        PDE_idx = torch.masked_select(all_idx, torch.logical_not(torch.isin(all_idx, torch.concat([BC_idx]))))  # Obtain remaining indices

        residual_pde1 = eta_u_x[PDE_idx] + eta_v_y[PDE_idx]
        residual_pde2 = eta_u[PDE_idx]*eta_u_x[PDE_idx] + eta_v[PDE_idx]*eta_u_y[PDE_idx] + eta_p_x[PDE_idx] - 0.002*(eta_u_xx[PDE_idx]+eta_u_yy[PDE_idx])
        residual_pde3 = eta_u[PDE_idx]*eta_v_x[PDE_idx] + eta_v[PDE_idx]*eta_v_y[PDE_idx] + eta_p_y[PDE_idx] - 0.002*(eta_v_xx[PDE_idx]+eta_v_yy[PDE_idx])
        residual_u_BCnoslip = eta_u[BC_idx] - U_col[BC_idx,0]
        residual_v_BCnoslip = eta_v[BC_idx] - U_col[BC_idx,1]
        residual_p_BCnoslip = eta_p[BC_idx] - U_col[BC_idx,2]

        loss_PDE1 = torch.mean(residual_pde1**2)
        loss_PDE2 = torch.mean(residual_pde2**2)
        loss_PDE3 = torch.mean(residual_pde3**2)
        loss_u_BCnoslip = torch.mean(residual_u_BCnoslip**2)
        loss_v_BCnoslip = torch.mean(residual_v_BCnoslip**2)
        loss_p_BCnoslip = torch.mean(residual_p_BCnoslip**2)

        loss_PDE = loss_PDE1 + loss_PDE2 + loss_PDE3 
        loss_BC = loss_u_BCnoslip + loss_v_BCnoslip + loss_p_BCnoslip

        xt.detach()

        return loss_PDE, loss_BC
    
    def calculate_data_loss(self):
        '''Calculate the loss function based on MLE depending on the mean function used.
        '''
        m_train_all = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])

        term2_U = 0.5 * torch.dot(self.U_data.to(torch.double) - m_train_all[:,0].to(torch.double), self.chol_decomp._cholesky_solve(self.U_data.unsqueeze(-1).to(torch.double) - m_train_all[:,0].unsqueeze(-1).to(torch.double)).squeeze(-1))
        term2_V = 0.5 * torch.dot(self.V_data.to(torch.double) - m_train_all[:,1].to(torch.double), self.chol_decomp._cholesky_solve(self.V_data.unsqueeze(-1).to(torch.double) - m_train_all[:,1].unsqueeze(-1).to(torch.double)).squeeze(-1))
        term2_P = 0.5 * torch.dot(self.P_data.to(torch.double) - m_train_all[:,2].to(torch.double), self.chol_decomp._cholesky_solve(self.P_data.unsqueeze(-1).to(torch.double) - m_train_all[:,2].unsqueeze(-1).to(torch.double)).squeeze(-1))

        loss =  term2_U + term2_V + term2_P  
            
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

                loss_PDE, loss_BC = self.calculate_PDE_loss(X_col, U_col)
                loss_data = self.calculate_data_loss()
                self.alpha, self.beta = self.compute_dynamic_weights(loss_PDE, loss_BC, loss_data)
                loss = loss_PDE # self.beta*loss_data + loss_PDE + self.alpha*loss_BC
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
        X_test, U_test = dataloader_to_tensor(test_dataloader)
        error_U = []
        error_V = []
        error_P = []
        error_Vel_mag = []

        C_Phi = self.kernel_phi(self.phi_unique)
        C_Y = self.kernel_y(self.y_unique)
        C = KroneckerProductLazyTensor(C_Phi, C_Y).to(torch.double)

        m_train = self.mean(self.train_inputs[0][:,:self.q], self.train_inputs[0][:,self.q:])

        C_inv_offset_U = C.cholesky()._cholesky_solve(self.U_data.unsqueeze(-1).to(torch.double) - m_train[:,0].unsqueeze(-1))
        C_inv_offset_V = C.cholesky()._cholesky_solve(self.V_data.unsqueeze(-1).to(torch.double) - m_train[:,1].unsqueeze(-1))
        C_inv_offset_P = C.cholesky()._cholesky_solve(self.P_data.unsqueeze(-1).to(torch.double) - m_train[:,2].unsqueeze(-1))

        for i in range(U_test.shape[0]):
            U_test_i = U_test[i,:,0].flatten()
            V_test_i = U_test[i,:,1].flatten()
            P_test_i = U_test[i,:,2].flatten()
            X_test_i = X_test[i,...]

            phi_col_unique = unique_rows_in_order(X_test_i[:,self.q:])
            y_col_unique = unique_rows_in_order(X_test_i[:,:self.q])

            C_yY = self.kernel_y(y_col_unique, self.y_unique).to(torch.double)
            C_Phiphi = self.kernel_phi(self.phi_unique, phi_col_unique).to(torch.double)
            
            m_col = self.mean(X_test_i[:,:self.q], X_test_i[:,self.q:])
                
            C_inv_offset_mat_U = C_inv_offset_U.reshape(C_Phiphi.shape[0], -1).t()
            C_inv_offset_mat_V = C_inv_offset_V.reshape(C_Phiphi.shape[0], -1).t()
            C_inv_offset_mat_P = C_inv_offset_P.reshape(C_Phiphi.shape[0], -1).t()
            U_pred = m_col[:,0] + (C_yY @ C_inv_offset_mat_U @ C_Phiphi).squeeze(-1)
            V_pred = m_col[:,1] + (C_yY @ C_inv_offset_mat_V @ C_Phiphi).squeeze(-1)
            P_pred = m_col[:,2] + (C_yY @ C_inv_offset_mat_P @ C_Phiphi).squeeze(-1)
            Vel_mag_pred = torch.sqrt(U_pred**2 + V_pred**2)
            Vel_mag_test = torch.sqrt(U_test_i**2 + V_test_i**2)

            e_U = np.mean(np.linalg.norm(U_pred.squeeze().cpu().detach().numpy()-U_test_i.cpu().squeeze().detach().numpy(), axis = -1)/np.linalg.norm(U_test_i.cpu().squeeze().detach().numpy(), axis = -1)) 
            error_U.append(e_U)

            e_V = np.mean(np.linalg.norm(V_pred.squeeze().cpu().detach().numpy()-V_test_i.cpu().squeeze().detach().numpy(), axis = -1)/np.linalg.norm(V_test_i.cpu().squeeze().detach().numpy(), axis = -1)) 
            error_V.append(e_V)

            e_P = np.mean(np.linalg.norm(P_pred.squeeze().cpu().detach().numpy()-P_test_i.cpu().squeeze().detach().numpy(), axis = -1)/np.linalg.norm(P_test_i.cpu().squeeze().detach().numpy(), axis = -1)) 
            error_P.append(e_P)

            e_Vel_mag = np.mean(np.linalg.norm(Vel_mag_pred.squeeze().cpu().detach().numpy()-Vel_mag_test.cpu().squeeze().detach().numpy(), axis = -1)/np.linalg.norm(Vel_mag_test.cpu().squeeze().detach().numpy(), axis = -1)) 
            error_Vel_mag.append(e_Vel_mag)

            if len(error_P) == 200:
                break

        return np.mean(error_Vel_mag)
    
    def compute_dynamic_weights(self, pde_loss, bc_loss, data_loss):
        lambdaa = 0.1
        params_to_update = [param for param in self.parameters() if param.requires_grad]
        
        delta_pde_teta = torch.autograd.grad(pde_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_pde_teta if p is not None]
        delta_pde_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))
        
        delta_bc_teta = torch.autograd.grad(self.alpha * bc_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values = [p.reshape(-1,).cpu().tolist() for p in delta_bc_teta if p is not None]
        delta_bc_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

        delta_data_teta = torch.autograd.grad(self.beta *data_loss, params_to_update,  retain_graph=True, allow_unused=True)
        values3 = [p.reshape(-1,).cpu().tolist() for p in delta_data_teta if p is not None]
        delta_data_teta_abs = torch.abs(torch.tensor([v for val in values3 for v in val]))

        temp3 = torch.max(delta_pde_teta_abs) / torch.mean(delta_data_teta_abs)
        temp = torch.max(delta_pde_teta_abs) / torch.mean(delta_bc_teta_abs)

        return (1.0 - lambdaa) * self.alpha + lambdaa * temp, (1.0 - lambdaa) * self.beta + lambdaa * temp3