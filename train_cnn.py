import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def train_cnn():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print("Building CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model (3 epochs)...")
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

    print("Saving model...")
    model.save('model.keras')
    print("Saved model.keras successfully.")

if __name__ == "__main__":
    train_cnn()
