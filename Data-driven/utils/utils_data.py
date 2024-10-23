import torch
import numpy as np
import random
import dill
import scipy.io as io
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import vtk as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import os

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load(fld):
    with open(fld, 'rb') as f:
        model = dill.load(f)
    return model

def unique_idx_in_order(A):
    A_np = np.array(A.cpu())
    _, unique_indices = np.unique(A_np, axis=0, return_index=True)
    
    return np.sort(unique_indices)

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


def get_data_Darcy(**tkwargs):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29, 
    # 25->17x17, 30 -> 15x15, 45 -> 10x10, 50 -> 9x9, 65 -> 7x7, 95 -> 5x5
    
    r_grid = 15
    s_grid = int(((421 - 1) / r_grid) + 1)

    r = 15
    s = int(((421 - 1) / r) + 1)

    ########## Training data set
    ntrain = 1000
    n_train = ntrain
    n_test = 200
    n_pca = min(ntrain, 200)
    PCA_flag = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_filename = "datasets/piececonst_r421_N1024_smooth1.mat"
    dataset_path = os.path.join(script_dir, dataset_filename)
    data = io.loadmat(dataset_path)

    A = data["coeff"][:n_train, ::r, ::r].astype(np.float64) / 12.0 #* 0.1 - 0.75
    U = data["sol"][:n_train, ::r_grid, ::r_grid].astype(np.float64) * 100

    U[:, 0, :] = 0
    U[:, -1, :] = 0
    U[:, :, 0] = 0
    U[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s_grid, dtype=np.float32))
    grids.append(np.linspace(0, 1, s_grid, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    A = A.reshape(n_train, s * s)

    if PCA_flag == True:
        pca = PCA(n_components=n_pca)
        A = pca.fit_transform(A)
        A = A[:, :n_pca]

    U = torch.from_numpy(U.reshape(n_train, s_grid * s_grid))

    X_train = torch.from_numpy(grid).unsqueeze(0).repeat(n_train,1,1).flatten(0,1)
    A_train = torch.from_numpy(A).unsqueeze(1).repeat(1,s_grid*s_grid,1).flatten(0,1)

    X_train = torch.hstack([X_train, A_train])
    U_train = U.flatten()

    ########## Test data set
    r_grid = 15
    s_grid = int(((421 - 1) / r_grid) + 1)
    dataset_filename = "datasets/piececonst_r421_N1024_smooth2.mat"
    dataset_path = os.path.join(script_dir, dataset_filename)
    data = io.loadmat(dataset_path)
    A = data["coeff"][-n_test:, ::r, ::r].astype(np.float64) / 12.0 #* 0.1 - 0.75
    U = data["sol"][-n_test:, ::r_grid, ::r_grid].astype(np.float64) * 100

    U[:, 0, :] = 0
    U[:, -1, :] = 0
    U[:, :, 0] = 0
    U[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s_grid, dtype=np.float32))
    grids.append(np.linspace(0, 1, s_grid, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    A = torch.from_numpy(A.reshape(n_test, s * s))

    if PCA_flag == True:
        A = pca.transform(A)[:, :n_pca]

    U_test = torch.from_numpy(U.reshape(n_test, s_grid*s_grid))
    X = torch.from_numpy(grid).unsqueeze(0).repeat(n_test,1,1)

    if PCA_flag == False:
        A = A.unsqueeze(1).repeat(1,s_grid*s_grid,1)
    else:
        A = torch.from_numpy(A).unsqueeze(1).repeat(1,s_grid*s_grid,1)
        
    X_test = torch.cat([X, A], dim=2)
    
    return X_train.to(**tkwargs), U_train.to(**tkwargs), X_test.to(**tkwargs), U_test.to(**tkwargs)

def get_data_Burgers(**tkwargs):
    dataset_path = os.path.join(script_dir, rf'datasets/burgers_data_R10.mat')
    data = io.loadmat(dataset_path)

    sub_u = 2**6
    sub_u_test = 2**6
    sub_a = 2**6
    s = 2**13 // sub_u

    U = data["u"][:, ::sub_u].astype(np.float64) # output: [N_ic, timesteps, x]
    A = data["a"][:, ::sub_a].astype(np.float64)

    n_train = 1000
    n_test = 200

    ##### Training data set
    U_grid = torch.from_numpy(U[:n_train,:])
    A_grid = torch.from_numpy(A[:n_train,:])
    
    X_grid = torch.linspace(0, 1, 2**13)[::sub_u, None].repeat(n_train,1)
    
    X_tensor = X_grid.flatten().unsqueeze(-1)
    A_tensor = A_grid.repeat_interleave(s, dim=0)

    X_train = torch.hstack([X_tensor, A_tensor])
    U_train = U_grid.flatten()

    ##### Test data set
    U = data["u"][:, ::sub_u_test].astype(np.float64) 
    A = data["a"][:, ::sub_a].astype(np.float64)

    s_test = 2**13 // sub_u_test
    U_grid_test = torch.from_numpy(U[-n_test:,:])
    X_grid = torch.linspace(0, 1, 2**13).repeat(n_test,1)[:,::sub_u_test].unsqueeze(-1)

    A_grid = torch.from_numpy(A[-n_test:,:]).unsqueeze(1).repeat(1,s_test,1)

    X_grid_test = torch.cat([X_grid, A_grid], dim=2)

    return X_train.to(**tkwargs), U_train.to(**tkwargs), X_grid_test.to(**tkwargs), U_grid_test.to(**tkwargs)

def get_data_Advection(**tkwargs):
    def get_data_train(filename, n):
        nx = 40
        nt = 40
        data = np.load(filename)
        u = data["u"].astype(np.float32)  # N x nt x nx

        u0 = u[:, 0, :]  # N x nx
        x = data["x"].astype(np.float32)[0, :][np.newaxis, :].repeat(u0.shape[0], axis=0)
        u = u[:, int(nt/2), :]

        X = torch.from_numpy(x.flatten()).unsqueeze(-1)
        U0 = torch.from_numpy(u0[:, np.newaxis, :].repeat(x.shape[1],axis=1)).flatten(0,1)
        U = torch.from_numpy(u.flatten())

        X_data = torch.hstack([X, U0])
        return X_data, U
    
    def get_data_test(filename, n):
        nx = 40
        nt = 40
        data = np.load(filename)
        u = data["u"].astype(np.float32)  # N x nt x nx

        u0 = u[:, 0, :]  # N x nx
        x = data["x"].astype(np.float32)[0, :][np.newaxis, :].repeat(u0.shape[0], axis=0)
        u = u[:, int(nt/2), :]

        X = torch.from_numpy(x).unsqueeze(-1)
        U0 = torch.from_numpy(u0[:, np.newaxis, :].repeat(x.shape[1],axis=1))
        U = torch.from_numpy(u)

        X_data = torch.cat([X, U0],axis=2)
        return X_data, U
    
    train_dataset_path = os.path.join(script_dir, rf'datasets/train_IC1.npz')
    test_dataset_path = os.path.join(script_dir, rf'datasets/test_IC1.npz')

    X_train, U_train = get_data_train(train_dataset_path, n = 1000)
    X_test, U_test = get_data_test(test_dataset_path, n = 200)

    return X_train.to(**tkwargs), U_train.to(**tkwargs), X_test.to(**tkwargs), U_test.to(**tkwargs)

def get_data_Structural(**tkwargs):
    n_train = 1000
    n_val = 250
    n_test = 20000

    inputs_path = os.path.join(script_dir, rf'datasets/StructuralMechanics_inputs.npy')
    outputs_path = os.path.join(script_dir, rf'datasets/StructuralMechanics_outputs.npy')        

    U0 = torch.from_numpy(np.load(inputs_path)).permute(2,0,1)
    U = torch.from_numpy(np.load(outputs_path)).permute(2,0,1)

    # Create a meshgrid for x and y coordinates
    x = np.linspace(0, 1, U0.shape[1])
    y = np.linspace(0, 1, U0.shape[1])

    # Create the grid
    Y, X = np.meshgrid(y, x)

    ################## Training data
    U_data = U[:n_train, :, :].flatten(1,2).flatten()
    U0_data = U0[:n_train,:,0].unsqueeze(1).repeat(1,41**2,1).flatten(0,1)
    XY_data = torch.hstack([torch.from_numpy(X.flatten()[:,np.newaxis]), torch.from_numpy(Y.flatten()[:,np.newaxis])]).repeat(n_train,1)
    
    X_train = torch.cat([XY_data, U0_data], dim=1)
    U_train = U_data

    ################## Test data
    U_data = U[-n_test:, :, :].flatten(1,2)
    U0_data = U0[-n_test:,:,0].unsqueeze(1).repeat(1,41**2,1)
    XY_data = torch.hstack([torch.from_numpy(X.flatten()[:,np.newaxis]), torch.from_numpy(Y.flatten()[:,np.newaxis])]).unsqueeze(0).repeat(n_test,1,1)
    
    X_test = torch.cat([XY_data, U0_data], dim=2)
    U_test = U_data

    mean_train = torch.mean(X_train, dim=0)
    std_train = torch.std(X_train, dim=0)
    mean_U_train = torch.mean(U_train, dim=0)
    std_U_train = torch.std(U_train, dim=0)

    # Standardize both X_train and X_test based on the statistics of X_train
    def standardize_tensor(tensor, mean, std):
        standardized_tensor = (tensor - mean) / std
        return standardized_tensor
    
    X_train = standardize_tensor(X_train, mean_train, std_train)
    X_test = torch.stack([(sample - mean_train) / std_train for sample in X_test])
    
    U_train = standardize_tensor(U_train, mean_U_train, std_U_train)

    train_dataset = TensorDataset(X_train.to(**tkwargs), U_train.to(**tkwargs))
    dataloader = DataLoader(train_dataset, batch_size = 41**2*100, shuffle=False)

    ################## Validation data
    U_data = U[n_train:n_train+n_val, :, :].flatten(1,2)
    U0_data = U0[n_train:n_train+n_val,:,0].unsqueeze(1).repeat(1,41**2,1)
    XY_data = torch.hstack([torch.from_numpy(X.flatten()[:,np.newaxis]), torch.from_numpy(Y.flatten()[:,np.newaxis])]).unsqueeze(0).repeat(n_val,1,1)
    
    X_val = torch.cat([XY_data, U0_data], dim=2)
    U_val = U_data 
    X_val = torch.stack([(sample - mean_train) / std_train for sample in X_val])

    return X_train, U_train, X_val, U_val, X_test, U_test, dataloader, mean_U_train, std_U_train


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

    s,  e = 0, 129 * 129
    ni, nj = 129, 129
    Coords_Block[...] = np.reshape(pntxyz[s:e, :2], (ni, nj, 2))
    Block[:, :, :2] = np.reshape(U[s:e, :2], (ni, nj, 2))
    Block[:, :, 2] = np.reshape(p[s:e], (ni, nj))

    Block_t = torch.from_numpy(Block).permute(2 , 0 , 1).unsqueeze(0)

    Block_ = torch.nn.functional.interpolate(Block_t, size=[120 , 120], scale_factor=None, mode='bilinear', align_corners=True, recompute_scale_factor=None, antialias=False)
    Block_ = Block_[0,...].permute(1 ,2 ,0).to("cuda")
    
    # You need to coarse the gridrm
    U =  Block_[: , : , 0]
    V = Block_[: , : , 1]
    P = Block_[: , : , 2]

    cell_size = 30 
    n_cell = int(120/cell_size)

    u_list = []
    v_list = []
    p_list = []
    u_list_extrapolation = []
    v_list_extrapolation = []
    p_list_extrapolation = []

    for j in range(0 , cell_size*(n_cell - 1) + 1, 3):
        for i in range(0 , cell_size*(n_cell - 1) + 1, 3):
            if i == 90:
                u_list_extrapolation.append(U[i:i+cell_size,j:j+cell_size])
                v_list_extrapolation.append(V[i:i+cell_size,j:j+cell_size])
                p_list_extrapolation.append(P[i:i+cell_size,j:j+cell_size])
            
            elif i <= 60:
                u_list.append(U[i:i+cell_size,j:j+cell_size])
                v_list.append(V[i:i+cell_size,j:j+cell_size])
                p_list.append(P[i:i+cell_size,j:j+cell_size])


    u = torch.cat([u.unsqueeze(0) for u in u_list] , dim = 0).to("cuda")
    v = torch.cat([v.unsqueeze(0) for v in v_list] , dim = 0).to("cuda")
    p = 0.05+torch.cat([p.unsqueeze(0) for p in p_list] , dim = 0).to("cuda")

    u_extrapolation = torch.cat([u.unsqueeze(0) for u in u_list_extrapolation] , dim = 0).to("cuda")
    v_extrapolation = torch.cat([v.unsqueeze(0) for v in v_list_extrapolation] , dim = 0).to("cuda")
    p_extrapolation = 0.05+torch.cat([p.unsqueeze(0) for p in p_list_extrapolation] , dim = 0).to("cuda")

    uvp = torch.cat([u.unsqueeze(-1) , v.unsqueeze(-1) , p.unsqueeze(-1)] , dim = -1) 
    uvp_extrapolation = torch.cat([u_extrapolation.unsqueeze(-1) , v_extrapolation.unsqueeze(-1) , p_extrapolation.unsqueeze(-1)] , dim = -1)

    uvp_interpolated = interpolate_grid(uvp.clone().permute(0,3,1,2))
    uvp_extrapolation_interpolated = interpolate_grid(uvp_extrapolation.clone().permute(0,3,1,2))

    N = uvp.shape[0]
    N_extrapolation = uvp_extrapolation.shape[0]

    uvp_flatten = uvp.view(N , -1 , uvp.shape[-1]).permute(0 , 2 ,1) 
    uvp_flatten_extrapolation = uvp_extrapolation.view(N_extrapolation , -1 , uvp_extrapolation.shape[-1]).permute(0 , 2 ,1)
    
    uvp_bc = uvp_interpolated 
    uvp_bc_extrapolation = uvp_extrapolation_interpolated

    x = np.linspace(0 , 1 , cell_size)
    y = np.linspace(0 , 1 , cell_size)
    X , Y = np.meshgrid(x, y)

    # Flatten X and T
    x_flat = X.reshape(-1)#flatten()
    y_flat = Y.reshape(-1)#.flatten()

    # Repeat and reshape x_flat and t_flat
    x = np.repeat(x_flat[np.newaxis , :], N, axis=0)  
    y = np.repeat(y_flat[np.newaxis , :], N, axis=0) 
    x_extrapolation = np.repeat(x_flat[np.newaxis , :], N_extrapolation, axis=0)
    y_extrapolation = np.repeat(y_flat[np.newaxis , :], N_extrapolation, axis=0)
    top = uvp_bc_extrapolation[:, :, 0, :]
    bottom = uvp_bc_extrapolation[:, :, -1, :]
    left = uvp_bc_extrapolation[:, :, :, 0]
    right = uvp_bc_extrapolation[:, :, :, -1]
    uvp_bc_flatten_extrapolation = torch.cat((top, right, bottom, left), dim=2)

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

    # Interpolation Data set
    ntrain = 500
    U_BC = uvp_bc_flatten[:ntrain, 0, :]
    V_BC = uvp_bc_flatten[:ntrain, 1, :]
    P_BC = uvp_bc_flatten[:ntrain, 2, :]
    U = uvp_flatten[:ntrain, 0, :]
    V = uvp_flatten[:ntrain, 1, :]
    P = uvp_flatten[:ntrain, 2, :]
    X = torch.from_numpy(x[:ntrain, :])
    Y = torch.from_numpy(y[:ntrain, :])
    
    X_train = torch.cat([X.flatten().unsqueeze(-1).to(**tkwargs), 
                         Y.flatten().unsqueeze(-1).to(**tkwargs), 
                         U_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs),
                         V_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs),
                         P_BC.unsqueeze(1).repeat(1,X.shape[1],1).flatten(0,1).to(**tkwargs)], dim = 1)
    UVP_train = torch.cat([U.flatten().unsqueeze(-1).to(**tkwargs), 
                           V.flatten().unsqueeze(-1).to(**tkwargs), 
                           P.flatten().unsqueeze(-1).to(**tkwargs)], dim = 1)
   
    # Test Data set
    ntest = 151

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
    
    # Extrapolation Data set
    U_BC = uvp_bc_flatten_extrapolation[:, 0, :]
    V_BC = uvp_bc_flatten_extrapolation[:, 1, :]
    P_BC = uvp_bc_flatten_extrapolation[:, 2, :]
    U = uvp_flatten_extrapolation[:, 0, :]
    V = uvp_flatten_extrapolation[:, 1, :]
    P = uvp_flatten_extrapolation[:, 2, :]
    X = torch.from_numpy(x_extrapolation[:, :])
    Y = torch.from_numpy(y_extrapolation[:, :])

    X_extrapolation = torch.cat([X.to(**tkwargs).unsqueeze(-1), 
                        Y.to(**tkwargs).unsqueeze(-1), 
                        U_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                        V_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs),
                        P_BC.unsqueeze(1).repeat(1,x.shape[1],1).to(**tkwargs)], dim = 2)
    UVP_extrapolation = torch.cat([U.to(**tkwargs).unsqueeze(-1), 
                          V.to(**tkwargs).unsqueeze(-1), 
                          P.to(**tkwargs).unsqueeze(-1)], dim = 2)
    
    return X_train.to(**tkwargs), UVP_train.to(**tkwargs), X_test.to(**tkwargs), UVP_test.to(**tkwargs), X_extrapolation.to(**tkwargs), UVP_extrapolation.to(**tkwargs), \
           uvp_bc[:ntrain, ...].permute(0,2,3,1).to(**tkwargs), uvp_bc[-ntest:, ...].permute(0,2,3,1).to(**tkwargs), uvp_bc_extrapolation.permute(0,2,3,1).to(**tkwargs)