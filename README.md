# Gaussian Processes for Operator Learning
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the paper [Operator Learning with Gaussian Processes](https://arxiv.org/abs/2409.04538), where we introduce a _general_ framework based on Gaussian Processes (GPs) for approximating single- or multi-output operators in either a purely data-driven context, or by using both data and physics. To achieve this, we cast operator learning as a regression problem, naturally suited for GP-baseed regression techniques:
![OperatorLearning_Diagram.pdf](https://github.com/user-attachments/files/17464680/OperatorLearning_Diagram.pdf)

The mean function of the GP can be set to zero or parameterized by a neural operator, and for each setting we develop a robust and scalable training strategy. These strategies rely on the assumption that both input and output functions are richly sampled at the same locations across samples. This allows us to 
1. Leverage the data structure together with the separability assumption on the kernel to formulate the GP's covariance matrix using the Kronecker product.
2. Initialize the kernel parameters such that they require little to no tuning.
While these strategies have proven effective for the problems tested in the aformentioned paper, we recommend that users explore alternative initialization settings, if they suspect that the above assumptions do not hold for their particular problem, using our insights and guidelines as a basis for developing strong baseline models.

Through a diverse set of numerical benchmarks, we demonstrate our method's scope, scalability, efficiency, and robustness. Our results show that, our method: 
1. Enhances the performance of a base neural operator by using it as the mean function of a GP.
2. Enables the construction of zero-shot data-driven models that can make accurate predictions without any prior training.
![ResultsTable](https://github.com/user-attachments/assets/420b0e6e-b0a4-4b03-8c09-f5b69fc74359)

## Requirements
Please ensure the following packages are installed with the specified versions. If you prefer to use Anaconda, the commands for creating an environment and installing these packages through its prompt are also provided:
- Python == 3.9.13: `conda create --name NN_CoRes python=3.9.13` and then activate the environment via `conda activate NN_CoRes`
- [PyTorch](https://github.com/pytorch/pytorch) == 1.12.0 & CUDA >= 11.3: `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch`
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) == 1.7.0: `conda install -c gpytorch gpytorch=1.7.0`
- [JAX](https://github.com/google/jax) == 0.4.25: `pip install jax==0.4.25 jaxlib==0.4.25 jaxtyping==0.2.25`
- Dill == 0.3.5.1: `pip install dill==0.3.5.1`
- Matplotlib == 3.5.3: `conda install -c conda-forge matplotlib=3.5.3`
- Tqdm >= 4.66.4 `pip install tqdm`

## Usage
After installing the above packages, you are all set to use our code. We provide two main files that demonstrate the application of our GP-based framework for solving the benchmark problems discussed in the paper.
You can test them by downloading the repo and running the following commands in your terminal:
- Burgers' equation: `python main_singleoutput.py --problem Burgers --parameter 0.003`
- Elliptic PDE: `python main_singleoutput.py --problem Elliptic --parameter 30`
- Eikonal equation: `python main_singleoutput.py --problem Eikonal --parameter 0.01`
- Lid-Driven Cavity: `python main_multioutput.py --problem LDC --parameter 5`

Alternatively, you can also simply run the files `main_singleoutput.py` or `main_multioutput.py` in your compiler.

You can use additional arguments to modify settings such as the architecture used in the mean function, optimizer, number of epochs, and more. Please refer to each main file for details.

## Contributions and Assistance
All contributions are welcome. If you notice any bugs, mistakes or have any question about the documentation, please report them by opening an issue on our GitHub page. Please make sure to label the issue according to the specific module or feature related to the problem.

## Citation
If you use this code or find our work interesting, please cite the following paper:
```bibtex
@article{mora2024operator,
  title={Operator Learning with Gaussian Processes},
  author={Mora, Carlos and Yousefpour, Amin and Hosseinmardi, Shirin and Owhadi, Houman and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2409.04538},
  year={2024}
}
