import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tf_layers

from .layers.linear import Linear
from .core.sequential import Sequential
from .losses.mse import MSELoss
from .optimizers.sgd import SGD

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    num_samples = 1000
    length = 64

    x_train_flat = x_train[:num_samples].reshape(num_samples, 784)
    x_train = x_train_flat[:, :length].reshape(num_samples, 1, length).astype("float32") / 255.0
    y_train_one_hot = tf.one_hot(y_train[:num_samples], depth=10).numpy()

    y_train_one_hot = y_train_one_hot.reshape(num_samples, 10, 1)

    in_channels = 1
    out_channels = 10
    kernel_size = 5
    stride = 1
    learning_rate = 0.01

    my_conv1d = Conv1D(in_channels, out_channels, kernel_size, stride=stride, padding="same")
    my_model = Sequential([my_conv1d])
    my_loss_fn = MSELoss()
    my_optimizer = SGD(my_model.layers, lr=learning_rate)
    
    tf_model = tf.keras.Sequential([
        tf_layers.Input(shape=(length, in_channels)),
        tf_layers.Conv1D(out_channels, kernel_size, strides=stride, padding='same', use_bias=True)
    ])
    tf_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    tf_loss_fn = tf.keras.losses.MeanSquaredError()

    w_init = my_conv1d.W.transpose(2, 1, 0)
    b_init = my_conv1d.b.flatten()
    tf_model.layers[0].set_weights([w_init, b_init])

    x_train_tf = x_train.transpose(0, 2, 1)
    y_train_tf = y_train_one_hot.transpose(0, 2, 1)

    print(f"{'Epoch':<10} | {'My Loss':<15} | {'TF Loss':<15} | {'Weight Diff':<15}")
    print("-" * 60)

    for epoch in range(1, 101):
        my_preds = my_model.forward(x_train)
        my_loss = my_loss_fn.forward(my_preds, y_train_one_hot)
        
        my_grad = my_loss_fn.backward()
        my_model.backward(my_grad)
        my_optimizer.step()

        with tf.GradientTape() as tape:
            tf_preds = tf_model(x_train_tf, training=True)
            tf_loss = tf_loss_fn(y_train_tf, tf_preds)
        
        grads = tape.gradient(tf_loss, tf_model.trainable_variables)
        tf_optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

        if epoch % 10 == 0 or epoch == 1:
            curr_tf_w = tf_model.layers[0].get_weights()[0].transpose(2, 1, 0)
            weight_diff = np.abs(my_conv1d.W - curr_tf_w).mean()
            print(f"{epoch:<10} | {my_loss:<15.6f} | {tf_loss.numpy():<15.6f} | {weight_diff:<15.2e}")

    my_final_out = my_model.forward(x_train).mean(axis=2) # (N, 10)
    my_accuracy = np.mean(np.argmax(my_final_out, axis=1) == y_train[:num_samples])

    tf_final_out = tf_model.predict(x_train_tf, verbose=0).mean(axis=1)
    tf_accuracy = np.mean(np.argmax(tf_final_out, axis=1) == y_train[:num_samples])
    
    print("\nFinal Results:")
    print(f"My Model Accuracy on MNIST: {my_accuracy:.4f}")
    print(f"TF Model Accuracy on MNIST: {tf_accuracy:.4f}")

if __name__ == "__main__":
    main()

# Epoch      | My Loss         | TF Loss         | Weight Diff
# ------------------------------------------------------------
# 1          | 0.100000        | 0.100000        | 1.71e-06
# 10         | 0.090767        | 0.099643        | 2.10e-05
# 20         | 0.089965        | 0.099261        | 4.78e-05
# 30         | 0.089913        | 0.098894        | 7.60e-05
# 40         | 0.089910        | 0.098541        | 1.05e-04
# 50         | 0.089910        | 0.098202        | 1.33e-04
# 60         | 0.089910        | 0.097877        | 1.62e-04
# 70         | 0.089910        | 0.097564        | 1.91e-04
# 80         | 0.089910        | 0.097264        | 2.19e-04
# 90         | 0.089910        | 0.096975        | 2.48e-04
# 100        | 0.089910        | 0.096698        | 2.76e-04

# Final Results:
# My Model Accuracy on MNIST: 0.1170
# TF Model Accuracy on MNIST: 0.1170