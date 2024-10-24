# Gaussian Processes for Operator Learning
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the paper [Operator Learning with Gaussian Processes](https://arxiv.org/abs/2409.04538), which introduces a **general framework** based on Gaussian Processes (GPs) for approximating single- or multi-output operators in either a purely data-driven context, or by using both data and physics. To achieve this, we cast operator learning as a regression problem, naturally suited for GP-based regression techniques:
![OperatorLearningDiagram](https://github.com/user-attachments/assets/e2afbbd3-601c-4a99-9863-2d5149b0e737)

The **mean function** of the GP can be set to **zero** or parameterized by a **neural operator**, and for each setting we develop a robust and scalable training strategy. These strategies rely on the assumption that both input and output functions are richly sampled at the same locations across samples, allowing us to: 
1. Leverage the data structure together with the separability assumption on the kernel to formulate the GP's covariance matrix using the **Kronecker product**.
2. **Initialize the kernel parameters** such that they require little to no tuning.

While these strategies have proven effective for the problems tested in the paper, we recommend that users explore alternative initialization settings, using our insights and guidelines as a basis for developing strong baseline models.

Through a diverse set of numerical benchmarks, we demonstrate our method's scope, scalability, efficiency, and robustness. Our results show that, our method: 
1. **Enhances the performance of a base neural operator** by using it as the mean function of a GP.
2. Enables the construction of **zero-shot** data-driven models that can make accurate predictions **without any prior training**.

![ResultsTable](https://github.com/user-attachments/assets/420b0e6e-b0a4-4b03-8c09-f5b69fc74359)

![Burgers_Darcy_Advection_Structural](https://github.com/user-attachments/assets/5c5814d1-0ac3-4735-9be7-4563aa68e39c)

## Requirements
To run the code, please install the following packages. If using Anaconda, you can create an environment and install the necessary packages as shown below:
- Python == 3.9.13: `conda create --name operatorGP python=3.9.13` and then activate the environment via `conda activate operatorGP`
- [PyTorch](https://github.com/pytorch/pytorch) == 1.12.0 & CUDA >= 11.3: `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch`
- [GPyTorch](https://github.com/cornellius-gp/gpytorch) == 1.7.0: `conda install -c gpytorch gpytorch=1.7.0`
- Dill == 0.3.5.1: `pip install dill==0.3.5.1`
- Tqdm >= 4.66.4: `pip install tqdm`
- Vtk: `pip install vtk`

## Usage
Once the packages are installed, you are ready to run the code. Download the repository and run the main files located in the *Data-driven* and *Physics-informed* folders. You can customize the architecture, kernel type, number of epochs, and other settings using the parser in the main files.

**Note**: We’ve observed that performance may vary slightly based on the hardware or software versions used, but this generally does not affect the order of magnitude of errors reported in the paper.

## Contributions and Assistance
All contributions are welcome! If you notice any bugs, mistakes or have any question about the documentation, please report them by opening an issue on our GitHub page. Please make sure to label the issue according to the specific module or feature related to the problem.

## Citation
If you use this code or find our work interesting, please cite the following paper:
```bibtex
@article{mora2024operator,
  title={Operator Learning with Gaussian Processes},
  author={Mora, Carlos and Yousefpour, Amin and Hosseinmardi, Shirin and Owhadi, Houman and Bostanabad, Ramin},
  journal={arXiv preprint arXiv:2409.04538},
  year={2024}
}
