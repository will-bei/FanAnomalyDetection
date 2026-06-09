import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def pixel_shuffle_native(x, block_size=8):
    # This assumes the input x has channels = C * (block_size^2)
    # Example: if output channels = 1, input channels must be 64
    x = tf.keras.layers.Reshape((-1, 8, block_size, block_size))(x) # Adjust dimensions to your needs
    x = tf.keras.layers.Permute((1, 3, 2, 4))(x)
    x = tf.keras.layers.Reshape((104, 16, 1))(x) # Target H, W, Channels
    return x

def create_tiny_anomaly_model(input_shape):
    model = models.Sequential(name="Pure_Acoustic_Classifier")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Arduino Stability Fix
    # GlobalAveragePooling2D reduces the [H, W, 128] output to a flat [128] vector.
    # It avoids the INT32 StridedSlice error entirely.
    model.add(layers.GlobalAveragePooling2D())

    # Dense Classifier Matrix
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

def create_tiny_autoencoder(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    bottleneck = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(x)

    # --- Bottleneck ---
    x = tf.keras.layers.Reshape((13 * 2 * 8,))(bottleneck)
    x = tf.keras.layers.Dense(13 * 2 * 8)(x) 
    x = tf.keras.layers.Reshape((13, 2, 8))(x)

    # --- Decoder (Using Resizing for stability) ---
    # 1. 13x2 -> 26x4
    x = tf.keras.layers.Resizing(26, 4, interpolation='nearest')(x)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)

    # 2. 26x4 -> 52x8
    x = tf.keras.layers.Resizing(52, 8, interpolation='nearest')(x)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)

    # 3. 52x8 -> 104x16
    x = tf.keras.layers.Resizing(104, 16, interpolation='nearest')(x)
    
    # Final output
    outputs = tf.keras.layers.Conv2D(1, 3, padding="same")(x)

    return tf.keras.Model(inputs, outputs)

    return tf.keras.Model(inputs, outputs)
def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model

def compile_autoencoder(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='mse'
    )
    return model