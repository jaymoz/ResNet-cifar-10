# ResNet for CIFAR-10 Classification

This repository contains the implementation of a Mini-Residual Neural Network in PyTorch, tailored for the CIFAR-10 image classification task. The ResNet architecture has been proven to be powerful for a wide range of vision tasks due to its ability to train deeper networks using residual connections.

## Key Features

- **Custom ResNet Architecture**: Implements a scaled-down version of the original ResNet suitable for the CIFAR-10 dataset with a 32x32 input size.
- **Residual Blocks**: Each block contains two layers of 3x3 convolutions with Batch Normalization and ReLU activation, learning residual mappings to combat the vanishing gradients problem in deep networks.
- **Downsampling**: Convolutional layers with stride 2 are used for downsampling within the network, effectively reducing the spatial dimensions while increasing the feature map depth.
- **Global Average Pooling**: Reduces each feature map to a single value, decreasing the number of parameters and computations needed in the network.
- **Dynamic Learning Rate Adjustment**: Employs a learning rate decay strategy, halving the learning rate every 20 epochs to ensure steady convergence.

## Usage

To use this Mini-ResNet model for CIFAR-10 classification:

1. Clone the repository.
2. Ensure you have PyTorch and torchvision installed.
3. Run the training script to start training the model.
4. After training, evaluate the model's performance on the CIFAR-10 test set.

## Model Structure

The network structure is as follows:
- Initial Convolutional Layer (3x3 conv, 16 filters)
- 2x Residual Blocks (3x3 conv, 16 filters)
- Downsampling Layer (3x3 conv, 32 filters, stride 2)
- 2x Residual Blocks (3x3 conv, 32 filters)
- Downsampling Layer (3x3 conv, 64 filters, stride 2)
- 2x Residual Blocks (3x3 conv, 64 filters)
- Global Average Pooling
- Fully Connected Layer (fc 10 classes)

## Training Details

The training process includes:
- Data augmentation with random horizontal flips and crops.
- Batch processing with a size of 100.
- Training for 25 epochs with a starting learning rate of 0.001.

The learning rate is adjusted dynamically during training, using a decay factor of 0.5 every 20 epochs.

## Results

The Mini-ResNet model achieves competitive accuracy on the CIFAR-10 test set, demonstrating the effectiveness of the residual learning paradigm on smaller networks and datasets.

