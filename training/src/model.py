import tensorflow as tf
from tensorflow.keras import layers, models

def create_tiny_anomaly_model(input_shape):
    """
    A pure, high-capacity CNN structure without BatchNormalization layers
    to prevent small-batch statistical blinding on imbalanced data.
    """
    model = models.Sequential(name="Pure_Acoustic_Classifier")
    
    # Block 1: Raw Feature Mapping
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Block 2: Deep Structure Learning
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Block 3: Aggressive Pattern Locking
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    
    # Dense Classifier Matrix
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    
    # Final Output
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model