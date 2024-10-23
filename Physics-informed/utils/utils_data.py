import torch
import numpy as np
import random
import scipy.io as io
from torch.utils.data import TensorDataset
import os
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk as vtk

# Get the directory of the current script
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def dataloader_to_tensor(dataloader):
    all_inputs = []
    all_labels = []

    for inputs, labels in dataloader:
        all_inputs.append(inputs)
        all_labels.append(labels)

    # Concatenate all batches into single tensors
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_inputs, all_labels

def get_data_BurgersDirichlet(**tkwargs):
    dataset_filename = "datasets/u_sol1BC_burger.mat"
    
    dataset_path = os.path.join(script_dir, dataset_filename)
    data = io.loadmat(dataset_path)

    U = data["sol"] 
    x = np.linspace(0, 1.0, U.shape[2])
    t = np.linspace(0, 1.0, U.shape[1])
    X, T = np.meshgrid(t,x)

    X, T, U = torch.from_numpy(X).to(**tkwargs), torch.from_numpy(T).to(**tkwargs), torch.from_numpy(U).to(**tkwargs)
    U0 = U[:,0:1,:].unsqueeze(-1).permute(0,1,3,2).repeat(1, X.shape[0], X.shape[1],1)

    ################# Kernel training data set #################
    n_train = 100
    n_col = 50
    n_test = 200
    n_u0 = 100

    u0_idx = torch.arange(0, X.shape[1], int(X.shape[1]/n_u0))
    n_points = 30
    x_idx_2 = torch.arange(0, X.shape[0], int(X.shape[0]/n_points))
    t_idx_2 = torch.arange(0, X.shape[1], int(X.shape[1]/n_points))
    n_x, n_t = X.shape[0], X.shape[1]
    selected_points = np.zeros((n_x, n_t), dtype=bool)
    selected_points[x_idx_2, 0] = True  # Top side
    selected_points[x_idx_2, -1] = True  # Bottom side
    selected_points[0, t_idx_2] = True  # Left side
    selected_points[-1, 0] = selected_points[-1, -1] = selected_points[0, 0] = selected_points[0,-1] = True  # Corner
    n_points = 10
    x_idx_2 = torch.arange(0, X.shape[0], int(X.shape[0]/(n_points+1)))[1:]
    t_idx_2 = torch.arange(0, X.shape[1], int(X.shape[1]/(n_points+1)))[1:-1]
    selected_points[x_idx_2[:, None], t_idx_2] = True
    selected_indices = np.argwhere(selected_points)

    X_grid = X[selected_indices[:, 0], selected_indices[:, 1]].repeat(n_train).unsqueeze(-1)
    T_grid = T[selected_indices[:, 0], selected_indices[:, 1]].repeat(n_train).unsqueeze(-1)
    U0_grid = U0[:n_train,:,:,:][:,:,:,u0_idx][:,selected_indices[:, 0], selected_indices[:, 1],:]
    kernel_inputs = torch.cat([X_grid, T_grid, U0_grid.flatten(start_dim=0, end_dim=1)], dim=1)
    U_grid = U[:n_train,:,:][:,selected_indices[:, 0], selected_indices[:, 1]]
    kernel_outputs = U_grid.flatten(start_dim=0, end_dim=1)
    kernel_train_dataset = TensorDataset(kernel_inputs, kernel_outputs)

    ##################### Training data set
    n_x = 100
    n_t = 100

    x_idx = torch.arange(0, X.shape[0], int(X.shape[0]/n_x))
    t_idx = torch.arange(0, X.shape[1], int(X.shape[1]/n_t))
    u0_idx = torch.arange(0, X.shape[1], int(X.shape[1]/n_u0))

    X_grid = X[x_idx,:][:,t_idx].unsqueeze(dim=0).repeat(n_col,1,1)
    T_grid = T[x_idx,:][:,t_idx].unsqueeze(dim=0).repeat(n_col,1,1)
     
    U0_grid = U0[400:400+n_col,:,:,:][:,x_idx,:,:][:,:,t_idx,:][:,:,:,u0_idx]
    U_grid = U[400:400+n_col,:,:][:,x_idx,:][:,:,t_idx]
    
    train_inputs = torch.cat([X_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), T_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), U0_grid.flatten(start_dim=1, end_dim=2)], dim=2)
    train_outputs = U_grid.flatten(start_dim=1, end_dim=2)
    train_dataset = TensorDataset(train_inputs, train_outputs)

    ##################### Test data set
    X_grid = X.unsqueeze(dim=0).repeat(n_test,1,1)
    T_grid = T.unsqueeze(dim=0).repeat(n_test,1,1)
    U0_grid = U0[-n_test:,:,:,:][:,:,:,u0_idx]
    U_grid = U[-n_test:,:,:] 
    
    test_inputs = torch.cat([X_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), T_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), U0_grid.flatten(start_dim=1, end_dim=2)], dim=2)
    test_outputs = U_grid.flatten(start_dim=1, end_dim=2)
    test_dataset = TensorDataset(test_inputs, test_outputs)

    return kernel_train_dataset, train_dataset, test_dataset

def get_bc(images):
    top_edge = images[:, 0, : , :]
    bottom_edge = images[:, -1, : , :]
    left_edge = images[:, :, 0 , :]
    right_edge = images[:, :, -1 , :]

    # Since the corners are included in both the top/bottom and left/right edges,
    # we remove them from the left and right edges to avoid duplication.
    left_edge = left_edge[:, 1:-1 , :]
    right_edge = right_edge[:, 1:-1  , :] 

    # Concatenate the edges. This will result in each row containing the outer pixels of one image.
    # The shape will be (7394, 168 , 3) because each edge has 43 pixels, and we remove 2*2 corners.
    outer_pixels = torch.cat([top_edge, right_edge, bottom_edge.flip(dims=[1]), left_edge.flip(dims=[1])], dim=1)
    return outer_pixels

def interpolate_grid(grid):
    M = grid.shape[-1] 
    for i in range(1 , M-1):
        N = M
        for j in range(1 , N-1):
            if i == j: grid[... , i , i] = (i)/N * grid[... , N-1 , N-1] + (N-i)/N * grid[... , 0 , 0]
            if j > i:
                m = j-i
                grid[... , i ,j] = (N-m-i)/(N-m) *grid[... , 0 , m] + (i)/(N-m) * grid[... , N-m-1 , N-1]
            if j < i:
                m = i-j
                grid[... , i ,j] = (N-m-j)/(N-m) *grid[... , m , 0] + (j)/(N-m) * grid[..., N-1 , N-m-1]

    return grid

def get_data_LDC(**tkwargs):
    dataset_filename = "datasets/square_cavity.vtk"
    dataset_path = os.path.join(script_dir, dataset_filename) 
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(dataset_path)
    reader.Update()
    vtkData = reader.GetOutput()

    p = vtk_to_numpy(vtkData.GetPointData().GetArray('p'))
    U = vtk_to_numpy(vtkData.GetPointData().GetArray('U'))
    pntxyz = vtk_to_numpy(vtkData.GetPoints().GetData())

    Block = np.zeros(((129, 129, 3)))
    Coords_Block = np.zeros((129, 129, 2))
    Coords_Block = np.zeros((129, 129, 2))

    s,  e = 0, 129 * 129
    ni, nj = 129, 129
    Coords_Block[...] = np.reshape(pntxyz[s:e, :2], (ni, nj, 2))
    Block[:, :, :2] = np.reshape(U[s:e, :2], (ni, nj, 2))
    Block[:, :, 2] = np.reshape(p[s:e], (ni, nj))

    Block_t = torch.from_numpy(Block).permute(2 , 0 , 1).unsqueeze(0)

    Block_ = torch.nn.functional.interpolate(Block_t, size=[120 , 120], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
    Block_ = Block_[0,...].permute(1 ,2 ,0).to("cuda")
    
    # You need to coarse the grid
    U =  Block_[: , : , 0]
    V = Block_[: , : , 1]
    P = Block_[: , : , 2]

    cell_size = 30 
    n_cell = int(120/cell_size)

    u_list = []
    v_list = []
    p_list = []

    for j in range(0 , cell_size*(n_cell -1)+1, 3):
        for i in range(0 , cell_size*(n_cell -1) , 3):
            u_list.append(U[i:i+cell_size,j:j+cell_size])
            v_list.append(V[i:i+cell_size,j:j+cell_size])
            p_list.append(P[i:i+cell_size,j:j+cell_size])

    u = torch.cat([u.unsqueeze(0) for u in u_list] , dim = 0).to("cuda")
    v = torch.cat([v.unsqueeze(0) for v in v_list] , dim = 0).to("cuda")
    p = 0.05+torch.cat([p.unsqueeze(0) for p in p_list] , dim = 0).to("cuda")

    uv =  torch.cat([u.unsqueeze(-1) , v.unsqueeze(-1) ] , dim = -1)
    uvp = torch.cat([u.unsqueeze(-1) , v.unsqueeze(-1) , p.unsqueeze(-1)] , dim = -1) #(7396, 43 , 43, 3)

    uvp_interpolated = interpolate_grid(uvp.clone().permute(0,3,1,2))

    N = uvp.shape[0]
    uvp_flatten = uvp.view(N , -1 , uvp.shape[-1]).permute(0 , 2 ,1) #(7396, 1849, 3) (7396, 3, 1849)
    uvp_bc = uvp_interpolated # get_bc(uvp).view(N , -1)#(78456 , 168*3).permute(0 , 2 ,1) #(7396, 168, 3)

    x = np.linspace(0 , 1 , cell_size)
    y = np.linspace(0 , 1 , cell_size)
    X , Y = np.meshgrid(x, y)

    # Flatten X and T
    x_flat = X.reshape(-1) #flatten()
    y_flat = Y.reshape(-1) #.flatten()

    # Repeat and reshape x_flat and t_flat
    x = np.repeat(x_flat[np.newaxis , :], N, axis=0)  # Shape: (batch_size, 10000)
    y = np.repeat(y_flat[np.newaxis , :], N, axis=0)  # Shape: (batch_size, 10000)

    # Shuffle data set
    top = uvp_bc[:, :, 0, :]
    bottom = uvp_bc[:, :, -1, :]
    left = uvp_bc[:, :, :, 0]
    right = uvp_bc[:, :, :, -1]
    
    uvp_bc_flatten = torch.cat((top, right, bottom, left), dim=2)

    seed = 42
    torch.manual_seed(seed)
    shuffled_indices = torch.randperm(uvp_bc.size(0))

    uvp_bc = uvp_bc[shuffled_indices,...]
    uvp_bc_flatten = uvp_bc_flatten[shuffled_indices,...]
    x = x[shuffled_indices,...]
    y = y [shuffled_indices,...]
    uvp_flatten = uvp_flatten[shuffled_indices,...]

    # Kernel Data set
    ntrain = 10

    U_BC = uvp_bc_flatten[:ntrain, 0, :]
    V_BC = uvp_bc_flatten[:ntrain, 1, :]
    P_BC = uvp_bc_flatten[:ntrain, 2, :]
    U = uvp_flatten[:ntrain, 0, :]
    V = uvp_flatten[:ntrain, 1, :]
    P = uvp_flatten[:ntrain, 2, :]
    X = torch.from_numpy(x[:ntrain, :])
    Y = torch.from_numpy(y[:ntrain, :])

    X_grid = X.reshape(-1, cell_size, cell_size)

    n_points = 30
    x_idx_2 = torch.arange(0, X_grid.shape[1], int(X_grid.shape[1]/n_points))
    t_idx_2 = torch.arange(0, X_grid.shape[2], int(X_grid.shape[2]/n_points))
    n_x, n_t = X_grid.shape[1], X_grid.shape[2]
    selected_points = np.zeros((n_x, n_t), dtype=bool)
    selected_points[x_idx_2, 0] = True  # Top side
    selected_points[x_idx_2, -1] = True  # Bottom side
    selected_points[0, t_idx_2] = True  # Left side
    selected_points[-1, t_idx_2] = True  # Right side
    selected_points[-1, 0] = selected_points[-1, -1] = selected_points[0, 0] = selected_points[0,-1] = True  # Corner
    n_points = 6
    x_idx_2 = torch.arange(0, X_grid.shape[1], int(X_grid.shape[1]/(n_points+1)))[1:-1]
    t_idx_2 = torch.arange(0, X_grid.shape[2], int(X_grid.shape[2]/(n_points+1)))[1:-1]
    selected_points[x_idx_2[:, None], t_idx_2] = True

    X_train = torch.cat([X.flatten().unsqueeze(-1).to(**tkwargs), 
                         Y.flatten().unsqueeze(-1).to(**tkwargs), 
                         U_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs),
                         V_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs),
                         P_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs)], dim = 1)
    UVP_train = torch.cat([U.flatten().unsqueeze(-1).to(**tkwargs), 
                           V.flatten().unsqueeze(-1).to(**tkwargs), 
                           P.flatten().unsqueeze(-1).to(**tkwargs)], dim = 1)
    
    kernel_dataset = TensorDataset(X_train, UVP_train)

    # Collocation Data set
    ncol = 100

    U_BC = uvp_bc_flatten[400:400+ncol, 0, :]
    V_BC = uvp_bc_flatten[400:400+ncol, 1, :]
    P_BC = uvp_bc_flatten[400:400+ncol, 2, :]
    U = uvp_flatten[400:400+ncol, 0, :]
    V = uvp_flatten[400:400+ncol, 1, :]
    P = uvp_flatten[400:400+ncol, 2, :]
    X = torch.from_numpy(x[400:400+ncol, :])
    Y = torch.from_numpy(y[400:400+ncol, :])

    X_train = torch.cat([X.unsqueeze(-1).to(**tkwargs), 
                         Y.unsqueeze(-1).to(**tkwargs), 
                         U_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                         V_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                         P_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs)], dim = 2)
    UVP_train = torch.cat([U.unsqueeze(-1).to(**tkwargs), 
                           V.unsqueeze(-1).to(**tkwargs), 
                           P.unsqueeze(-1).to(**tkwargs)], dim = 2)
    
    train_dataset = TensorDataset(X_train, UVP_train)

    # Test Data set
    ntest = 200

    U_BC = uvp_bc_flatten[-ntest:, 0, :]
    V_BC = uvp_bc_flatten[-ntest:, 1, :]
    P_BC = uvp_bc_flatten[-ntest:, 2, :]
    U = uvp_flatten[-ntest:, 0, :]
    V = uvp_flatten[-ntest:, 1, :]
    P = uvp_flatten[-ntest:, 2, :]
    X = torch.from_numpy(x[-ntest:, :])
    Y = torch.from_numpy(y[-ntest:, :])

    X_test = torch.cat([X.to(**tkwargs).unsqueeze(-1), 
                        Y.to(**tkwargs).unsqueeze(-1), 
                        U_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                        V_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                        P_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs)], dim = 2)
    
    UVP_test = torch.cat([U.to(**tkwargs).unsqueeze(-1), 
                          V.to(**tkwargs).unsqueeze(-1), 
                          P.to(**tkwargs).unsqueeze(-1)], dim = 2)
    
    test_dataset = TensorDataset(X_test, UVP_test)
       
    return kernel_dataset, train_dataset, test_dataset