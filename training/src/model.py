import tensorflow as tf
from tensorflow.keras import layers, models

def create_tiny_anomaly_model(input_shape):
    """
    Constructs a memory-efficient CNN suitable for TFLite Micro deployment.
    Input shape expected: (Time_Steps, MFCC_Features, 1)
    """
    model = models.Sequential(name="TinyML_Acoustic_Classifier")
    
    # Layer 1: Downsample time and space with small kernel sizes
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1)) # Small dropout to prevent overfitting
    
    # Layer 2: Deeper features, reduced dimensions
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    
    # Flatten to dense layer
    model.add(layers.Flatten())
    
    # Keep dense layer very small to control RAM footprint
    model.add(layers.Dense(16, activation='relu'))
    
    # Binary Output: Normal vs Abnormal
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

if __name__ == "__main__":
    # Test footprint calculations assuming 1-second windows: ~101 time steps x 13 MFCCs
    sample_shape = (101, 13, 1) 
    test_model = create_tiny_anomaly_model(sample_shape)
    test_model.summary()