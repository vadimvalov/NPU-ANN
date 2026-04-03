# NPU-ANN: Artificial Neural Network Framework (CNN)

This project is a modular framework for building Artificial Neural Networks from scratch using Python and NumPy. It was developed as part of the ANN course at NPU. The platform supports various layer types, activation functions, and backpropagation without relying on autograd features from libraries like PyTorch or TensorFlow.

## 1. Getting Started

To set up the environment and run the project, follow these steps:

1. **Navigate to the source directory:**
```
cd src
```
2. **Install dependencies:**

The project requires NumPy and TensorFlow (used for cross-validation).

    
    pip install -r requirements.txt
    
3. **Run the project:**

Launch the main script as a module to execute the training demonstration and validation:

```
python -m task1.main
```

## 2. Project Structure

Every commit is a separate and atomic part of a task, so during the demonstration we would jump from commit to commit and see how it becames fully developed framework from scratch. 

1. **Linear layer**:

To go to this exact commit use:

```
git checkout 4e2968773c3b4e07ce35cdc31252fbf8516c8c2c
```

Here we can see the implementation of a linear layer with forward and backward passes. Only three files are present, which are `main.py`, `layers/linear.py` and `core/layer.py`.

That is enough to demonstrate the proper work of a linear layer and a layer itself. 

Layer class is an abstract class that defines the interface for all layers in the network. It has two methods: `forward` and `backward`.

Linear layer is a layer that performs a linear transformation of the input data. It has two parameters: weights and biases. 

Main has an input, which is a $2 \times 3$ feature matrix that undergoes a linear transformation ($XW + b$) to produce a $2 \times 4$ output. During backpropagation, the input gradient is calculated as grad_out @ W.T, the weight gradient as x.T @ grad_out, and the bias gradient as the sum of output gradients across the batch.

2. **ReLU activation**:

To go to this exact commit use:

```
git checkout 50d30ff4e864123749ee2e9c69c6106b7b78c73c
```

Here we can see the implementation of the first activation function, which is basically the ReLU activation. Changes in this commit touches two files, which are `relu.py` and `main.py` for testing purposes

ReLU is a non-linear activation function that is defined as $f(x) = max(0, x)$. It is taking only positive values.

Testing in on the main shows the change in GRAD b from [[2, 2, 2, 2]] to [[1, 2, 0, 2]] occurs because the ReLU layer acts as a gate during backpropagation.

3. **Sequential container**:

To go to this exact commit use:

```
git checkout 71c69d22d35aca6a971cfb023c560cb54a8ab70a
```

Here we can see the implementation of sequential model, which is a container for layers. It allows us to create a model with multiple layers and activation functions, which are gonna be trained together in a linear sequence.

Changes from 

``` 
    layer = Linear(3, 4)
    relu = ReLU()
```

to

```
    model = Sequential([
        Linear(3, 4),
        ReLU()
    ])
```

are present. It actually does not change nothing, but makes the code more readable and wouldn't be messed up with large-scale models.

4. **Tanh and Sigmoid activations**:

To go to this commit use:

```
git checkout 1e2550c5ea3f44fbecd7fd8c5dd904cd0be770fc
```

We just implemented two additional activations. ReLU is not differentiable at 0, so we need to use something else for the hidden layers. Thus, we are making Tanh and Sigmoid. Tanh uses np.tanh(x), and Sigmoid uses 1 / (1 + np.exp(-x)) math func. 

As simple as that

5. **MSE**:

To go to this commit use:

```
git checkout 5f13d15d9f08a7780cbd6def59a3208ee046b837
```

Here we are demonstating the MSE, which is Mean Squared Error, and is calculated as (y_pred - y_true)^2 / n. As an output in main we can notice the LOSS, which is equal to 0.9706321916512981 with random seed equals to 42. At this point we understand that the model has some error and is not accurate at all.

6. **Optimizer**:

To go to this commit use:

```
git checkout d550f6cda101699337347a8de0dfd89dc6e57215
```

We understood that the model is dumb, we need to learn it to be more accurate. Thus, we are implementing the SGD optimizer, which is Stochastic Gradient Descent. 

Here we demonstrates how it changes the loss after one iteration from 0.9722058667327826 to 0.9640448544347593. A progress begins, it becomes smarter.

7. **First train loop for 100 epochs**:

To go here use:

```
git checkout 636167da9c9a92f13f31f67addac72456f64dee6
```

1 iteration is good, 101 iteration is better. Fully use everything we've got right now: activations, sequential and layers. Forward and backward propagation etc. 

8. **Conv1D**: 

To go here use:

```
git checkout d35f5314b8d3d26746556e28640eb38364ab0370
```

We understood how to work with Linear, now lets continue with Conv1D. It is basically the same as Linear, but with a sliding window. 

Conv1D is a layer that performs a 1D convolution of the input data. It has two parameters: weights and biases. 

Here we can see the implementation of Conv1D layer and its usage in a main function. 

9. **Conv2D**:

To go here use:

```
git checkout 14a5f4555693341e34f37f949425ccd01cf14b76
```

Conv1D is not as interesting as Conv2D. Conv2D is a layer that performs a 2D convolution of the input data. It has two parameters: weights and biases also.

Works, learns, getting better and better. Cool

10. **Comparing our Linear and TensorFlow's linear**:

To go here use:

```
git checkout 4dce10ca882087afe0ab96d1c967b83a0fc5da13
```

Further commits would just test our model on MNIST dataset and compare it with TensorFlow's model.

11. **Comparing our Conv1D and TensorFlow's Conv1D**:

To go here use:

```
git checkout 840148d355904488474495417645840059664797
```

12. **Comparing our Conv2D and TensorFlow's Conv2D**:

To go here use:

```
git checkout d1591f5d6ac4ea5371ba23e6ebcca3fb37f90737
```


