from lmgp_pytorch.preprocessing import standard 
from lmgp_pytorch.preprocessing import setlevels
from sklearn.model_selection import train_test_split
import torch

def train_test_split_normalizeX(
    X,
    y,
    test_size=None,
    shuffle=True,
    stratify=None,
    qual_index_val = {},
    return_mean_std = False,
    set_levels = False
):
    # Finding the quant index from qual index
    qual_index = list(qual_index_val.keys())
    all_index = set(range(X.shape[-1]))
    quant_index = list(all_index.difference(qual_index))
    if set_levels:
        # This will assign levels to categorical evenif the levels are strings
        X = setlevels(X, qual_index = qual_index)
    # Split test and train
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
        test_size= test_size, shuffle= shuffle, stratify=stratify)
    # Standard
    Xtrain, Xtest, mean_train, std_train = standard(Xtrain = Xtrain, 
        quant_index = quant_index, Xtest = Xtest)

    ytrain = torch.tensor(ytrain)
    ytest = torch.tensor(ytest)

    if return_mean_std:
        return Xtrain, Xtest, ytrain, ytest, mean_train, std_train

    return Xtrain, Xtest, ytrain, ytest
    