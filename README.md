# BTP (Bachelor Thesis Project) - VGG Implementation

## Introduction

This project implements the VGG neural network architecture for the purpose of research into sparsity penalties. The code is provided in a Jupyter Notebook (`btp-vgg.ipynb`) available in this GitHub repository.

## Setup

To run the code, follow these steps:

1. Clone this GitHub repository to your local machine:

    ```bash
    git clone <repository_url>
    ```

2. Open the `btp-vgg.ipynb` file in a Jupyter Notebook environment.

3. Optionally, upload the notebook to Kaggle for execution (if required).

## Execution

1. Run the cells in the notebook sequentially. 

2. Customize the hyperparameters as needed. These include:

   - Number of epochs: Modify the `epochs` variable.
   - Learning rate: Adjust the `learning_rate` parameter.
   - Sparsity penalties:
        - For L1 penalty: Use the `updateBN()` function.
        - For Lp penalty: Use the `updateLpBN(a)` function, where `a` is the value of `p`.
        - For TL1 penalty: Use the `updateTL1BN(a)` function, where `a` is the hyper-parameter.
   
3. Monitor the training process and evaluate the model's performance.

## Notes

- Ensure you have sufficient computational resources available for training the VGG model, especially if running on large datasets or with many epochs.
- It is recommended to run the notebook in an environment with GPU acceleration for faster training times.
- Save the trained model weights and any important results for further analysis.

## References

- VGG: Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.
- [Kaggle](https://www.kaggle.com/): Platform for data science and machine learning competitions.