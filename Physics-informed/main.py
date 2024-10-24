import torch
from torch.utils.data import DataLoader
from GP.models.GP_BurgersDirichlet import GP_BurgersDirichlet
from GP.models.GP_LDC import GP_LDC
from utils.utils_data import get_data_BurgersDirichlet, get_data_LDC, set_seed
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(description='GP for Operator Learning')
    
    # Problem 
    parser.add_argument("--problem", type = str, default = 'LDC') # Use 'BurgersDirichlet' or 'LDC'
    parser.add_argument("--N", type = int, default = '50') # Number of training samples

    # Kernel type (use GPyTorch naming)
    parser.add_argument("--kernel_phi", type = str, default = 'MaternKernel') # Use 'MaternKernel' or 'RBFKernel'
    parser.add_argument("--kernel_y", type = str, default = 'MaternKernel') # Use 'MaternKernel' or 'RBFKernel'
    
    # Optimization settings
    parser.add_argument("--epochs", type = int, default = 500)

    # Random seed
    parser.add_argument("--randomseed", type = int, default = 1)

    # Dtype, device
    parser.add_argument("--tkwargs", type = dict, default = {"dtype": torch.float, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")})

    args = parser.parse_args()    
    
    return args

def main(options):
    ############################### 1. Generate Data ############################################
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fld = os.path.join(script_dir, rf'Checkpoints/PI-GP_{options.epochs}epochs')

    if options.problem == 'BurgersDirichlet':
        kernel_train_dataset, train_dataset, test_dataset = get_data_BurgersDirichlet(options.N, **options.tkwargs)
    elif options.problem == 'LDC':
        kernel_train_dataset, train_dataset, test_dataset = get_data_LDC(options.N, **options.tkwargs)

    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=True)

    ############################### 2. Build Model ##############################################
    if options.problem == 'BurgersDirichlet':
        model = GP_BurgersDirichlet(train_dataset = kernel_train_dataset,
                                    kernel_phi = 'MaternKernel',
                                    kernel_y = 'MaternKernel',
                                    **options.tkwargs)
        
    elif options.problem == 'LDC':
        model = GP_LDC(train_dataset = kernel_train_dataset,
                        kernel_phi = 'MaternKernel',
                        kernel_y = 'MaternKernel',
                        **options.tkwargs)
    
    ############################### 3.a Train and (Optionally) Save Model ####################################
    model.fit(train_dataloader = train_dataloader, 
              test_dataloader = test_dataloader, 
              fld = fld, 
              num_iter = options.epochs, 
              **options.tkwargs)
    # model.save(fld = fld) # If instead you want to load a saved model, use model = load(fld = '...')

    ############################### 4. Evaluate Model ###########################################
    l2_error = model.evaluate_error(test_dataloader)
    print(f'Test relative L2 error: {l2_error * 100:.3f}%')

if __name__ == '__main__':
    options = get_parser()
    set_seed(options.randomseed)
    main(options)
