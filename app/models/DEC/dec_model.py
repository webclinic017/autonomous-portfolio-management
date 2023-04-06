import tensorflow as tf
import pandas as pd
import numpy as np

# Define the model architecture
class DECModel(tf.keras.Model):
    def __init__(self, n_clusters, input_shape):
        super(DECModel, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(2000, activation='relu'),
            tf.keras.layers.Dense(10, activation=None)
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2000, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(input_shape, activation='linear')
        ])
        
        self.n_clusters = n_clusters
        

    def compile(self, optimizer, loss_fn, loss=None, learning_rate_scheduler=None, **kwargs):
        super(DECModel, self).compile(**kwargs)  # Pass any additional arguments to the parent class
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss = loss  # Add this line to handle the 'loss' argument
        self.learning_rate_scheduler = learning_rate_scheduler

        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
        
    def cluster_layer(self, inputs):
        inputs = tf.cast(inputs, tf.float32)  # Add this line to cast inputs to float32
        self.encoder.weights[-1] = tf.cast(self.encoder.weights[-1], tf.float32)  # Add this line to cast encoder weights to float32
        reshaped_weights = tf.reshape(self.encoder.weights[-1], [1, 10, 1])
        q = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - reshaped_weights), axis=2)
        q = 1.0 / (1.0 + q / self.n_clusters)
        q = q ** (2.0 / 3.0)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q
    
    
def dec_loss(x, x_reconstructed, q, q_target):
    reconstruction_loss = tf.keras.losses.mean_squared_error(x, x_reconstructed)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)  # Ensure scalar value
    clustering_loss = tf.keras.losses.KLD(q, q_target)
    clustering_loss = tf.reduce_mean(clustering_loss)  # Ensure scalar value
    return reconstruction_loss + clustering_loss

# Create a function to generate target distribution for clustering loss
def target_distribution(q):
    weight = q ** 2 / tf.reduce_sum(q, axis=0)
    denominator = tf.reduce_sum(weight, axis=1, keepdims=True)
    return weight / denominator

def train_dec_model(model, dataset, n_epochs, early_stopping_patience):
    # Create a variable to store the best loss found
    best_loss = float('inf')
    # Create a counter to track the number of epochs without improvement
    no_improvement_counter = 0

    for epoch in range(n_epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                x = batch
                x_reconstructed = model(x)
                q = model.cluster_layer(x)
                q_target = target_distribution(q)  # Calculate q_target here
                loss = dec_loss(x, x_reconstructed, q, q_target)  # Pass q_target to the loss function

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Update learning rate if a scheduler is provided
        if model.learning_rate_scheduler:
            model.optimizer.learning_rate = model.learning_rate_scheduler(epoch)
        print("Loss shape:", loss.shape)
        print("Loss content:", loss.numpy())
        print("Epoch {}: Loss {:.3f}".format(epoch+1, float(loss.numpy())))

        # Early stopping logic
        if loss.numpy() < best_loss:
            best_loss = loss.numpy()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break