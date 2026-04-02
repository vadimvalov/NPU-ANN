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
    x_train = x_train[:num_samples].reshape(num_samples, 784).astype("float32") / 255.0
    y_train_one_hot = tf.one_hot(y_train[:num_samples], depth=10).numpy()

    in_features = 784
    out_features = 10
    learning_rate = 0.1

    my_linear = Linear(in_features, out_features)
    my_model = Sequential([my_linear])
    my_loss_fn = MSELoss()
    my_optimizer = SGD(my_model.layers, lr=learning_rate)

    tf_model = tf.keras.Sequential([
        tf_layers.Input(shape=(in_features,)),
        tf_layers.Dense(out_features, use_bias=True)
    ])
    tf_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    tf_loss_fn = tf.keras.losses.MeanSquaredError()

    tf_model.layers[0].set_weights([my_linear.W, my_linear.b.flatten()])

    print(f"{'Epoch':<10} | {'My Loss':<15} | {'TF Loss':<15} | {'Weight Diff':<15}")
    print("-" * 60)

    for epoch in range(1, 101):
        my_preds = my_model.forward(x_train)
        my_loss = my_loss_fn.forward(my_preds, y_train_one_hot)
        
        my_grad = my_loss_fn.backward()
        my_model.backward(my_grad)
        my_optimizer.step()

        with tf.GradientTape() as tape:
            tf_preds = tf_model(x_train, training=True)
            tf_loss = tf_loss_fn(y_train_one_hot, tf_preds)
        
        grads = tape.gradient(tf_loss, tf_model.trainable_variables)
        tf_optimizer.apply_gradients(zip(grads, tf_model.trainable_variables))

        if epoch % 10 == 0 or epoch == 1:
            current_tf_w = tf_model.layers[0].get_weights()[0]
            weight_diff = np.abs(my_linear.W - current_tf_w).mean()
            
            print(f"{epoch:<10} | {my_loss:<15.6f} | {tf_loss.numpy():<15.6f} | {weight_diff:<15.2e}")

    my_final_preds = np.argmax(my_model.forward(x_train), axis=1)
    tf_final_preds = np.argmax(tf_model.predict(x_train, verbose=0), axis=1)
    
    my_accuracy = np.mean(my_final_preds == y_train[:num_samples])
    tf_accuracy = np.mean(tf_final_preds == y_train[:num_samples])

    print("\nFinal Results:")
    print(f"My Model Accuracy: {my_accuracy:.4f}")
    print(f"TF Model Accuracy: {tf_accuracy:.4f}")

if __name__ == "__main__":
    main()