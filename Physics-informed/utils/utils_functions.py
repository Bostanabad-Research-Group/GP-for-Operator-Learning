import torch
import numpy as np
import random
import dill
import scipy.io as io
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


import os
from scipy import io

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load(fld):
    with open(fld, 'rb') as f:
        model = dill.load(f)
    return model

def get_data_darcy():
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29
    r = 15
    s = int(((421 - 1) / r) + 1)

    ########## Training data set
    n_train = 1000
    n_test = 200
    n_pca = 10
    PCA_flag = True

    data = io.loadmat(r"C:\Users\98car\Downloads\piececonst_r421_N1024_smooth1.mat")
    A = data["coeff"][:n_train, ::r, ::r].astype(np.float64) #* 0.1 - 0.75
    U = data["sol"][:n_train, ::r, ::r].astype(np.float64) * 100

    U[:, 0, :] = 0
    U[:, -1, :] = 0
    U[:, :, 0] = 0
    U[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    A = A.reshape(n_train, s * s)

    if PCA_flag == True:
        pca = PCA(n_components=A.shape[-1])
        A = pca.fit_transform(A)
        n_pca = 100 # np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0]
        A = A[:, :n_pca]

    U = torch.from_numpy(U.reshape(n_train, s * s))

    X_train = torch.from_numpy(grid).unsqueeze(0).repeat(n_train,1,1).flatten(0,1)
    A_train = torch.from_numpy(A).unsqueeze(1).repeat(1,s*s,1).flatten(0,1)

    X_train = torch.hstack([X_train, A_train])
    U_train = U.flatten()

    ########## Test data set
    data = io.loadmat(r"C:\Users\98car\Downloads\piececonst_r421_N1024_smooth2.mat")
    A = data["coeff"][:n_test, ::r, ::r].astype(np.float64) #* 0.1 - 0.75
    U = data["sol"][:n_test, ::r, ::r].astype(np.float64) * 100

    U[:, 0, :] = 0
    U[:, -1, :] = 0
    U[:, :, 0] = 0
    U[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    A = torch.from_numpy(A.reshape(n_test, s * s))

    if PCA_flag == True:
        A = pca.transform(A)[:, :n_pca]

    U_test = torch.from_numpy(U.reshape(n_test, s * s))
    X = torch.from_numpy(grid).unsqueeze(0).repeat(n_test,1,1)
    A = torch.from_numpy(A).unsqueeze(1).repeat(1,s*s,1)
    X_test = torch.cat([X, A], dim=2)
    
    return X_train, U_train, X_test, U_test

def get_data_burgers_dirichlet(ntrain, **tkwargs):
    dataset_filename = "u_sol1BC_burger.mat"
    dataset_path = os.path.join(script_dir, dataset_filename)
    data = io.loadmat(dataset_path)

    U = data["sol"] 
    x = np.linspace(0, 1.0, U.shape[2])
    t = np.linspace(0, 1.0, U.shape[1])
    X, T = np.meshgrid(t,x)

    X, T, U = torch.from_numpy(X).to(**tkwargs), torch.from_numpy(T).to(**tkwargs), torch.from_numpy(U).to(**tkwargs)
    U0 = U[:,0:1,:].unsqueeze(-1).permute(0,1,3,2).repeat(1, X.shape[0], X.shape[1],1)
    
    ################# Training data set #################
    n_ICs_kernel = ntrain
    n_u0 = 30
    u0_idx = torch.arange(0, X.shape[1], int(X.shape[1]/n_u0))
    n_points = 30
    x_idx_2 = torch.arange(0, X.shape[0], int(X.shape[0]/n_points))
    t_idx_2 = torch.arange(0, X.shape[1], int(X.shape[1]/n_points))
    n_x, n_t = X.shape[0], X.shape[1]
    selected_points = np.zeros((n_x, n_t), dtype=bool)
    # selected_points[x_idx_2, 0] = True  # Top side
    # selected_points[x_idx_2, -1] = True  # Bottom side
    # selected_points[0, t_idx_2] = True  # Left side
    # selected_points[-1, 0] = selected_points[-1, -1] = selected_points[0, 0] = selected_points[0,-1] = True  # Corner
    n_points = 30
    x_idx_2 = torch.arange(0, X.shape[0], int(X.shape[0]/(n_points)))
    t_idx_2 = torch.arange(0, X.shape[1], int(X.shape[1]/(n_points)))
    selected_points[x_idx_2[:, None], t_idx_2] = True
    selected_indices = np.argwhere(selected_points)

    # plt.imshow(selected_points, cmap='binary', interpolation='nearest')
    # plt.title('Boolean Grid')
    # plt.colorbar(label='True/False')
    # plt.show()
    
    X_grid = X[selected_indices[:, 0], selected_indices[:, 1]].repeat(n_ICs_kernel).unsqueeze(-1)
    T_grid = T[selected_indices[:, 0], selected_indices[:, 1]].repeat(n_ICs_kernel).unsqueeze(-1)
    U0_grid = U0[:n_ICs_kernel,:,:,:][:,:,:,u0_idx][:,selected_indices[:, 0], selected_indices[:, 1],:]
    X_train = torch.cat([X_grid, T_grid, U0_grid.flatten(start_dim=0, end_dim=1)], dim=1)
    U_grid = U[:n_ICs_kernel,:,:][:,selected_indices[:, 0], selected_indices[:, 1]]
    U_train = U_grid.flatten(start_dim=0, end_dim=1)

    ###################### Test data set
    X_grid = X.unsqueeze(dim=0).repeat(U.shape[0]-(n_ICs_kernel),1,1)
    T_grid = T.unsqueeze(dim=0).repeat(U.shape[0]-(n_ICs_kernel),1,1)
    U0_grid = U0[-(U.shape[0]-(n_ICs_kernel)):,:,:,:][:,:,:,u0_idx]
    U_grid = U[-(U.shape[0]-(n_ICs_kernel)):,:,:] 
    
    X_test = torch.cat([X_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), T_grid.flatten(start_dim=1, end_dim=2).unsqueeze(-1), U0_grid.flatten(start_dim=1, end_dim=2)], dim=2)
    U_test = U_grid.flatten(start_dim=1, end_dim=2)

    return X_train, U_train, X_test, U_test

def get_data_burgers_kernel( ntrain, **tkwargs):
    #dataset_path = r"G:\My Drive\PhD\My projects\NN-CoRes_OperatorLearning\Datasets\Burgers\Burgers"
    dataset_filename = "burgers_data_R10.mat"
    dataset_path = os.path.join(script_dir, dataset_filename)
    data = io.loadmat(dataset_path)

    sub_u = 2**6
    sub_a = 2**6
    s = 2**13 // sub_u

    U = data["u"][:, ::sub_u].astype(np.float64) # output: [N_ic, timesteps, x]
    A = data["a"][:, ::sub_a].astype(np.float64)
    
    # Define the RBF kernel function
    def rbf_kernel(x1, x2, sigma):
        diff = x1 - x2
        return torch.exp(-10**sigma*torch.sum(diff**2))

    n_train = ntrain
    n_test = 200
    n_pca = 10
    PCA_flag = False
    
    # Generate a random permutation of indices
    # perm = np.random.permutation(U.shape[0])
    # U = U[perm, :]
    # A = A[perm, :]

    ##### Training data set
    U_grid = torch.from_numpy(U[:n_train,:])

    if PCA_flag == True:
        pca = PCA(n_components = n_pca)
        A_grid = torch.from_numpy(pca.fit_transform(A[:n_train,:]))
        A_grid_test = torch.from_numpy(pca.transform(A[n_train:n_train+n_test,:]))
    else:
        A_grid = torch.from_numpy(A[:n_train,:])
        A_grid_test = torch.from_numpy(A[n_train:n_train+n_test,:])
    
    X_grid = torch.linspace(0, 1, 2**13)[::sub_u, None].repeat(n_train,1)
    
    X_tensor = X_grid.flatten().unsqueeze(-1)
    A_tensor = A_grid.repeat_interleave(s, dim=0)

    X_train = torch.hstack([X_tensor, A_tensor])
    U_train = U_grid.flatten()
    
    train_dataset = TensorDataset(torch.cat([torch.linspace(0, 1, 2**13)[::sub_u, None].unsqueeze(0).repeat(n_train,1,1), A_grid.unsqueeze(1).repeat(1,s,1)], dim=2), U_grid)
    dataloader = DataLoader(train_dataset, batch_size = 10, shuffle=True)

    plot_hist = False
    if plot_hist == True:
        # Values of omega
        omegas = [-3,-2,-1]

        # Plot histograms for each value of omega
        plt.figure(figsize=(12, 8))
        for omega in omegas:
            print(omega)
            # Calculate pairwise correlation
            distances = torch.norm(A_grid[:, None, :] - A_grid, dim=2)
            correlations = torch.exp(-10**omega * distances**2)
            # Plot histogram
            plt.hist(correlations.triu(diagonal=1).flatten(), bins=50, alpha=0.4, label=f'omega = {omega}')

        plt.xlabel('Pairwise Correlation')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pairwise Correlations for Different Length Scales')
        plt.legend()
        plt.show()

    subspace_check = False
    if subspace_check == True:
        # Compute the pseudoinverse of the basis matrix
        basis_pseudoinv = torch.pinverse(A_grid)

        # Project the new vectors onto the subspace spanned by the basis
        projection = torch.mm(A_grid_test, basis_pseudoinv)

        # Compute the reconstructed vectors from the projection
        reconstructed = torch.mm(projection, A_grid)

        # Compute the difference between the original vectors and their reconstructions
        difference = A_grid_test - reconstructed

        # Compute the norms of the differences
        norms = torch.norm(difference, dim=1)

        # Threshold to determine if the vectors belong to the subspace (adjust as needed)
        threshold = 1e-6

        # Check if the norms are smaller than the threshold
        belong_to_subspace = norms < threshold

        # Compute the extent to which the vectors belong to the subspace
        extent = 1 - norms / torch.norm(A_grid_test, dim=1)

        # Print the results
        print("Belongs to subspace:", belong_to_subspace)
        print("Extent of belonging to subspace:", extent)

    check_hypercube = True
    if check_hypercube:
        def check_within_hypercube(existing_vectors, new_vectors):
            # Compute minimum and maximum values for each component of existing vectors
            min_values, _ = torch.min(existing_vectors, dim=0)
            max_values, _ = torch.max(existing_vectors, dim=0)
            
            # Count the number of vectors outside the hypercube
            count_outside = 0
            for vector in new_vectors:
                if torch.any(vector < min_values) or torch.any(vector > max_values):
                    count_outside += 1
            return count_outside

        print(f"{check_within_hypercube(A_grid, A_grid_test)} new vectors lie outside the hypercube.")

    ##### Test data set
    U_grid_test = torch.from_numpy(U[-n_test:,:])
    X_grid = torch.linspace(0, 1, 2**13).repeat(n_test,1)[:,::sub_u].unsqueeze(-1)

    if PCA_flag == True:
        A_grid = torch.from_numpy(pca.transform(A[-n_test:,:])).unsqueeze(1).repeat(1,s,1)
    else:
        A_grid = torch.from_numpy(A[-n_test:,:]).unsqueeze(1).repeat(1,s,1)
    
    # U_test = U_grid.flatten()
    # X_tensor = X_grid.flatten().unsqueeze(-1)
    # A_tensor = A_grid.repeat_interleave(X_grid.shape[1], dim=0)

    X_grid_test = torch.cat([X_grid, A_grid], dim=2)

    return X_train, U_train, X_grid_test, U_grid_test, dataloader

def get_data_burgers(**tkwargs):
    #dataset_path = r"G:\My Drive\PhD\My projects\NN-CoRes_OperatorLearning\Datasets\Burgers\Burgers"
    dataset_path = r"C:\Users\98car\Downloads\burgers_data_R10.mat"
    data = io.loadmat(dataset_path)

    sub_u = 2**6
    sub_a = 2**6
    s = 2**13 // sub_u

    U = data["u"][:, ::sub_u].astype(np.float64) # output: [N_ic, timesteps, x]
    A = data["a"][:, ::sub_a].astype(np.float64)

    ################################ PCA ###################################
    # _,S,V = torch.svd(torch.from_numpy(A))
    # k = 10 
    # principal_components = V[:, :k]
    # projected_data = torch.from_numpy(A) @ principal_components
    # total_variance = torch.sum(torch.square(S))  # Total variance is sum of squares of singular values
    # explained_variance = torch.sum(torch.square(S[:k]))  # Explained variance is sum of squares of the first k singular values
    # explained_variance_ratio = explained_variance / total_variance
    # print("Explained Variance Ratio:", explained_variance_ratio.item())
    # PCA = projected_data
    
    ########################################################################
    n_x = 128
    n_a = 128
    n_train = 1000
    n_test = 200
    PCA_flag = False
    
    # Generate a random permutation of indices
    perm = np.random.permutation(n_train + n_test)
    U = U[perm, :]
    A = A[perm, :]
    PCA = PCA[perm, :]

    x_idx = torch.arange(0, U.shape[1], int(A.shape[1]/n_x))
    a_idx = torch.arange(0, A.shape[1], int(A.shape[1]/n_a))

    ##### Training data set
    U_grid = torch.from_numpy(U[:, x_idx][:n_train,:]).to(**tkwargs)
    if PCA_flag == True:
        mean = np.mean(PCA[:n_train+n_test,:].numpy(), axis=0)
        std_dev = np.std(PCA[:n_train+n_test,:].numpy(), axis=0)
        PCA = (PCA - mean) / std_dev
        A_grid = PCA[:n_train,:].to(**tkwargs)
    else:
        mean = np.mean(A[:n_train+n_test,:], axis=0)
        std_dev = np.std(A[:n_train+n_test,:], axis=0)
        A = (A - mean) / std_dev
        A_grid = torch.from_numpy(A[:, a_idx][:n_train,:]).to(**tkwargs)

    X_grid = torch.linspace(0, 1, 2**13)[::sub_u, None].repeat(n_train,1).to(**tkwargs)
    U_train = U_grid.flatten()
    X_tensor = X_grid.flatten().unsqueeze(-1)
    A_tensor = A_grid.repeat_interleave(n_x, dim=0)

    X_train = torch.hstack([X_tensor, A_tensor])

    ##### Test data set
    n_x = 128
    U_grid_test = torch.from_numpy(U[:, x_idx][n_train:n_train+n_test,:]).to(**tkwargs)
    X_grid = torch.linspace(0, 1, 2**13).repeat(n_test,1)[:,::sub_u].to(**tkwargs).unsqueeze(-1)

    if PCA_flag == True:
        A_grid = PCA[n_train:n_train+n_test,:].to(**tkwargs).unsqueeze(1).repeat(1,n_x,1)
    else:
        A_grid = torch.from_numpy(A[:, a_idx][n_train:n_train+n_test,:]).to(**tkwargs).unsqueeze(1).repeat(1,n_x,1)
    

    # U_test = U_grid.flatten()
    # X_tensor = X_grid.flatten().unsqueeze(-1)
    # A_tensor = A_grid.repeat_interleave(X_grid.shape[1], dim=0)

    X_grid_test = torch.cat([X_grid, A_grid], dim=2)

    return X_train, U_train, X_grid_test, U_grid_test

def get_data_burgers_old():
    dataset_path = r"G:\My Drive\PhD\My projects\NN-CoRes_OperatorLearning\Datasets\Burgers\Burgers"
    data = io.loadmat(dataset_path)
    U = data["output"] # output: [N_ic, timesteps, x]
    x = np.linspace(0, 1, U.shape[2])
    t = np.linspace(0, 1, U.shape[1])
    X, T = np.meshgrid(t,x)

    return X,T,U

def get_data_advectionII():
    inputs_path = r"G:\My Drive\PhD\My projects\NN-CoRes_OperatorLearning_NLL_V2\Datasets\AdvectionII\Advection_inputs.npy"
    outputs_path = r"G:\My Drive\PhD\My projects\NN-CoRes_OperatorLearning_NLL_V2\Datasets\AdvectionII\Advection_outputs.npy"

    U0 = np.load(inputs_path)
    U = np.load(outputs_path)
    X = np.linspace(0.0, 1.0, U.shape[0])[:, np.newaxis]

    return X, U0, U

def get_data_structural():
    inputs_path = r"C:\Users\98car\Downloads\StructuralMechanics_inputs.npy"
    outputs_path = r"C:\Users\98car\Downloads\StructuralMechanics_outputs.npy"

    U0 = np.load(inputs_path)
    U = np.load(outputs_path)

    # Create a meshgrid for x and y coordinates
    x = np.linspace(0, 1, U0.shape[0])
    y = np.linspace(0, 1, U0.shape[0])

    # Create the grid
    Y, X = np.meshgrid(y, x)

    return X, Y, U0, U

def get_data_structural_kernel(**tkwargs):
    inputs_path = r"C:\Users\98car\Downloads\StructuralMechanics_inputs.npy"
    outputs_path = r"C:\Users\98car\Downloads\StructuralMechanics_outputs.npy"

    U0 = torch.from_numpy(np.load(inputs_path)).permute(2,0,1)
    U = torch.from_numpy(np.load(outputs_path)).permute(2,0,1)

    # Create a meshgrid for x and y coordinates
    x = np.linspace(0, 1, U0.shape[1])
    y = np.linspace(0, 1, U0.shape[1])

    # Create the grid
    Y, X = np.meshgrid(y, x)

    n_train = 2500
    n_test = 200

    ################## Training data
    U_data = U[:n_train, :, :].flatten(1,2).flatten()
    U0_data = U0[:n_train,:,0].unsqueeze(1).repeat(1,41**2,1).flatten(0,1)
    XY_data = torch.hstack([torch.from_numpy(X.flatten()[:,np.newaxis]), torch.from_numpy(Y.flatten()[:,np.newaxis])]).repeat(n_train,1)
    
    X_train = torch.cat([XY_data, U0_data], dim=1)
    U_train = U_data
    

    # plt.scatter(X_train[:(41)*(41),0], X_train[:(41)*(41),1], c=U_train[:(41)*(41)])
    # plt.colorbar(label='u') 
    # plt.show()

    ################## Test data
    U_data = U[n_train:n_train+n_test, :, :].flatten(1,2)
    U0_data = U0[n_train:n_train+n_test,:,0].unsqueeze(1).repeat(1,41**2,1)
    XY_data = torch.hstack([torch.from_numpy(X.flatten()[:,np.newaxis]), torch.from_numpy(Y.flatten()[:,np.newaxis])]).unsqueeze(0).repeat(n_test,1,1)
    
    X_test = torch.cat([XY_data, U0_data], dim=2)
    U_test = U_data
    # train_dataset = TensorDataset(X_train, U_train)

    mean_train = torch.mean(X_train, dim=0)
    std_train = torch.std(X_train, dim=0)

    # Standardize both X_train and X_test based on the statistics of X_train
    def standardize_tensor(tensor, mean, std):
        standardized_tensor = (tensor - mean) / std
        return standardized_tensor
    
    X_train = standardize_tensor(X_train, mean_train, std_train)
    X_test = torch.stack([(sample - mean_train) / std_train for sample in X_test])

    train_dataset = TensorDataset(X_train, U_train)
    dataloader = DataLoader(train_dataset, batch_size = 1000, shuffle=False)

    return X_train, U_train, X_test, U_test, dataloader