# Task 1. Image classification + OOP

## Overview of the solution
In this task I worked on classification problem for [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset. 
To begin with, the data were loaded using the tools of the tensorflow library and were normalized.
The dataset is balanced, so we didn't need any stratification technique. 

In the modelling part I tried 3 different models: Random Forest, Feed Forward Neural Network and Convolutional Neural Network.
Training process is customizable via `.json` file. All models have unified interface with the same format of inputs and outputs.


Source code is written in Python 3.11. All code is written in OOP style with [SOLID](https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design) principles.

## Classes
### 1. MnistClassifierInterface
This class define the contract between all classifiers. It has two abstract methods: `train` and `predict` that are unified for all models.

### 2. RandomForestModel
With default setting builds 100 decision trees with `max_depth` of 20. It flattens X's from 2D to 1D in `train` and `predict`. This model is imported from scikit-learn.  

### 3. NNClassifier
It processes input data with a shape of 784 (e.g., flattened 28x28 images) and outputs predictions for 10 classes using a softmax activation function. The model consists of two dense layers:
- First layer: 256 units, ReLU activation, 20% dropout
- Second layer: 128 units, ReLU activation, 20% dropout

The network is optimized using the Adam optimizer with a learning rate of 0.001 and is trained using categorical cross-entropy loss. The training process includes accuracy as a performance metric and runs for 10 epochs with a batch size of 32, using 20% of the data for validation.

### 4. CNNClassifier
CNNClassifier is a convolutional neural network for classifying 28×28 grayscale images into 10 classes. It has:

- Two convolutional layers:
  - First: 16 filters, 3×3 kernel, ReLU activation, same padding
  - Second: 32 filters, 3×3 kernel, ReLU activation, same padding
- A 2×2 pooling layer for downsampling
- A dense layer with 128 units and ReLU activation
- A softmax output layer with 10 units
- The model uses the Adam optimizer, sparse categorical crossentropy loss, a batch size of 32, trains for 10 epochs, and reserves 20% of the data for validation.

### 5. MnistClassifier
It is wrapper class that defines algorithm by input parameter (`'rf'`, `'nn'`, `'cnn'`) and takes path to `models_config.json` file

## Project structure
- `models` - a folder with models classes and configuration file.
  -  `conv_nn.py` - a file with Convolutional NN model class.
  -  `feed_forward_nn.py` - a file with Feed-Forward NN model class.
  -  `random_forest.py` - a file with Random Forest model class.
  -  `models_config.json` - a models parameters configuration file.
- `models_interface.py` - a file with interface class.
- `mnist_classifier.py` - a file with class-wrapper.
- `demo_notebook` - notebook with demostration of the work.
- `requirements.txt` - text file with required libraries.

## Results 
| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| RF    | 0.97      | 0.97   | 0.97     |
| NN    | 0.98      | 0.98   | 0.98     |
| CNN   | 0.99      | 0.99   | 0.99     |

## How to set-up a project?
### 1. **Clone the repository**
   Clone this repository to your local machine using:

   ```bash
git clone https://github.com/znak314/Winstars_AI.git
   ```
### 2. **Create a virtual environment**

   ```bash
cd task_1
python -m venv .venv
.venv\Scripts\activate     # Windows 
source .venv/bin/activate  # Linux
   ```
### 3. **Install all necessary libraries**
   Install all dependencies by using:

   ```bash
   pip install -r requirements.txt
   ```
