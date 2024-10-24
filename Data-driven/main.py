import torch
from GP.models.GP_Burgers import GP_Burgers
from GP.models.GP_Darcy import GP_Darcy
from GP.models.GP_Advection import GP_Advection
from GP.models.GP_Structural import GP_Structural
from GP.models.GP_LDC import GP_LDC

from utils.utils_data import get_data_Burgers, get_data_Darcy, get_data_Advection, get_data_Structural, get_data_LDC, set_seed, load
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='GP for Operator Learning')
    
    # Problem 
    parser.add_argument("--problem", type = str, default = 'Burgers') # Use 'Burgers', 'Darcy', 'Advection', 'Structural' or 'LDC'
    parser.add_argument("--N", type = int, default = '1000') # Number of training samples

    # Kernel and mean type (use GPyTorch naming for kernels)
    parser.add_argument("--kernel_phi", type = str, default = 'MaternKernel') # Use 'MaternKernel' or 'RBFKernel'
    parser.add_argument("--kernel_y", type = str, default = 'MaternKernel') # Use 'MaternKernel' or 'RBFKernel'
    parser.add_argument("--mean_type", type = str, default = 'FNO') # Use 'zero', 'DeepONet' or 'FNO'
    
    # Number of epochs
    parser.add_argument("--epochs", type = int, default = 2000)

    # Random seed
    parser.add_argument("--randomseed", type = int, default = 1)

    # Dtype, device
    parser.add_argument("--tkwargs", type = dict, default = {"dtype": torch.float32, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")})

    args = parser.parse_args()    
    
    return args

def main(options):
    ############################### 1. Generate Data ############################################
    if options.problem == 'Burgers':
        X_train, U_train, X_test, U_test = get_data_Burgers(options.N, **options.tkwargs)
    elif options.problem == 'Darcy':
        X_train, U_train, X_test, U_test = get_data_Darcy(options.N, **options.tkwargs)
    elif options.problem == 'Advection':
        X_train, U_train, X_test, U_test = get_data_Advection(options.N, **options.tkwargs)
    elif options.problem == 'Structural':
        X_train, U_train, X_val, U_val, X_test, U_test, _, mean_U_train, std_U_train = get_data_Structural(options.N, **options.tkwargs)
    elif options.problem == 'LDC':
        X_train, U_train, X_test, U_test, X_extrapolation, U_extrapolation, U_BC_FNO_train, U_BC_FNO_test, U_BC_FNO_extrapolation = get_data_LDC(options.N, **options.tkwargs)

    ############################### 2. Build Model ##############################################
    if options.problem == 'Burgers':
        model = GP_Burgers(X_data = X_train,
                           U_data = U_train,
                           kernel_y = options.kernel_y,
                           kernel_phi = options.kernel_phi,
                           mean_type = options.mean_type,
                           **options.tkwargs)
        
    elif options.problem == 'Darcy':
        model = GP_Darcy(X_data = X_train,
                         U_data = U_train,
                         kernel_y = options.kernel_y,
                         kernel_phi = options.kernel_phi,
                         mean_type = options.mean_type,
                         **options.tkwargs)
        
    elif options.problem == 'Advection':
        model = GP_Advection(X_data = X_train, 
                             U_data = U_train,
                             mean_type = options.mean_type,
                             **options.tkwargs)
        
    elif options.problem == 'Structural':
        model = GP_Structural(X_data = X_train, 
                              U_data = U_train,
                              X_val = X_val, 
                              U_val = U_val,
                              mean_type = options.mean_type,
                              mean_U_train = mean_U_train, 
                              std_U_train = std_U_train,
                              **options.tkwargs)
    
    elif options.problem == 'LDC':
        model = GP_LDC(X_data = X_train, 
                       U_data = U_train,
                       UVP_BC_FNO_train = U_BC_FNO_train,
                       UVP_BC_FNO_test = U_BC_FNO_test,
                       UVP_BC_FNO_extrapolation = U_BC_FNO_extrapolation,
                       mean_type = options.mean_type,
                       **options.tkwargs)
    
    ############################### 3.a Train and (Optionally) Save Model ####################################
    model.fit(X_test, U_test, num_iter = options.epochs)
    # model.save(fld = fld) # If instead you want to load a saved model, use model = load(fld = '...')

    ############################### 4. Evaluate Model ###########################################
    if options.problem == 'LDC':
        rl2_error = model.evaluate_error(X_test, U_test, extrapolation = False) # For extrapolation error, use X_extrapolation, UVP_extrapolation and set extrapolation = True
    else:
        rl2_error = model.evaluate_error(X_test, U_test)

    print(f'Test relative L2 error: {rl2_error * 100:.3f}%')

if __name__ == '__main__':
    options = get_parser()
    set_seed(options.randomseed)
    main(options)
