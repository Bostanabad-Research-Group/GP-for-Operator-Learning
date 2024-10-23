
import numpy as np
import torch


def standard(Xtrain, quant_index, Xtest = None):
    if len(quant_index) == 0:
        return Xtrain
    
    temp = Xtrain[..., quant_index]
    if type(temp) != torch.Tensor:
        temp = temp.astype(float)

    mean_xtrain = temp.mean(axis = 0)
    std_xtrain = temp.std(axis = 0)
    temp=(temp - mean_xtrain)/std_xtrain
    Xtrain[..., quant_index] = temp
    if type(Xtrain) == np.ndarray:
        Xtrain = torch.from_numpy(Xtrain)
    if Xtest is None:
        return Xtrain,mean_xtrain, std_xtrain
    else:
        temp2 = Xtest[..., quant_index]
        if type(temp2) != torch.Tensor:
            temp2 = temp2.astype(float)
        temp2 = (temp2 - mean_xtrain)/std_xtrain
        Xtest[..., quant_index] = temp2
        if type(Xtest) == np.ndarray:
            Xtest = torch.from_numpy(Xtest)
        return Xtrain, Xtest, mean_xtrain, std_xtrain

