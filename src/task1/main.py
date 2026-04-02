import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tf_layers

from .layers.conv2d import Conv2D
from .core.sequential import Sequential
from .losses.mse import MSELoss
from .optimizers.sgd import SGD

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    (x_train_raw, y_train_raw), _ = tf.keras.datasets.mnist.load_data()
    
    num_samples = 100
    size = 14
    
    x_train = x_train_raw[:num_samples, :size, :size].reshape(num_samples, 1, size, size).astype("float32") / 255.0
    y_train_one_hot = tf.one_hot(y_train_raw[:num_samples], depth=10).numpy()
    
    y_train_target = np.repeat(y_train_one_hot[:, :, np.newaxis], size, axis=2)
    y_train_target = np.repeat(y_train_target[:, :, :, np.newaxis], size, axis=3)

    in_channels = 1
    out_channels = 10 
    kernel_size = 3
    stride = 1
    learning_rate = 0.01

    my_conv2d = Conv2D(in_channels, out_channels, kernel_size, stride=stride, padding="same")
    my_model = Sequential([my_conv2d])
    my_loss_fn = MSELoss()
    my_optimizer = SGD(my_model.layers, lr=learning_rate)

    tf_model = tf.keras.Sequential([
        tf_layers.Input(shape=(size, size, in_channels)),
        tf_layers.Conv2D(out_channels, kernel_size, strides=stride, padding='same', use_bias=True)
    ])
    tf_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    tf_loss_fn = tf.keras.losses.MeanSquaredError()

    w_init = my_conv2d.W.transpose(2, 3, 1, 0)
    b_init = my_conv2d.b.flatten()
    tf_model.layers[0].set_weights([w_init, b_init])

    x_train_tf = x_train.transpose(0, 2, 3, 1)
    y_train_tf = y_train_target.transpose(0, 2, 3, 1)

    print(f"{'Epoch':<10} | {'My Loss':<15} | {'TF Loss':<15} | {'Weight Diff':<15}")
    print("-" * 60)

    for epoch in range(1, 101):
        my_preds = my_model.forward(x_train)
        my_loss = my_loss_fn.forward(my_preds, y_train_target)
        
        my_grad = my_loss_fn.backward()
        my_model.backward(my_grad)
        my_optimizer.step()

        with tf.GradientTape() as tape:
            tf_preds = tf_model(x_train_tf, training=True)
            tf_loss = tf_loss_fn(y_train_tf, tf_preds)
        
        grads = tape.gradient(tf_loss, tf_model.trainable_variables)
        tf_optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

        if epoch % 10 == 0 or epoch == 1:
            curr_tf_w = tf_model.layers[0].get_weights()[0].transpose(3, 2, 0, 1)
            weight_diff = np.abs(my_conv2d.W - curr_tf_w).mean()
            print(f"{epoch:<10} | {my_loss:<15.6f} | {tf_loss.numpy():<15.6f} | {weight_diff:<15.2e}", flush=True)

    my_final_out = my_model.forward(x_train).mean(axis=(2, 3))
    my_accuracy = np.mean(np.argmax(my_final_out, axis=1) == y_train_raw[:num_samples])
    
    tf_final_out = tf_model.predict(x_train_tf, verbose=0).mean(axis=(1, 2))
    tf_accuracy = np.mean(np.argmax(tf_final_out, axis=1) == y_train_raw[:num_samples])
    
    print("\nFinal Results:")
    print(f"My Model Accuracy: {my_accuracy:.4f}")
    print(f"TF Model Accuracy: {tf_accuracy:.4f}")

if __name__ == "__main__":
    main()

# Epoch      | My Loss         | TF Loss         | Weight Diff
# ------------------------------------------------------------
# 1          | 0.100079        | 0.100079        | 2.22e-10
# 10         | 0.099668        | 0.099668        | 4.69e-10
# 20         | 0.099230        | 0.099230        | 6.31e-10
# 30         | 0.098810        | 0.098810        | 8.82e-10
# 40         | 0.098408        | 0.098408        | 9.47e-10
# 50         | 0.098023        | 0.098023        | 1.02e-09
# 60         | 0.097654        | 0.097654        | 1.17e-09
# 70         | 0.097300        | 0.097300        | 1.21e-09
# 80         | 0.096962        | 0.096962        | 1.28e-09
# 90         | 0.096637        | 0.096637        | 1.39e-09
# 100        | 0.096327        | 0.096327        | 1.49e-09

# Final Results:
# My Model Accuracy: 0.2300
# TF Model Accuracy: 0.2300