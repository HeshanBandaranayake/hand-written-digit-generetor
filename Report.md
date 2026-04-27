# MNIST GAN Digit Generator Project

## Overview

This project implements a Generative Adversarial Network (GAN) trained on the MNIST dataset to generate handwritten digits. A web interface is connected to the trained model using a Flask backend, allowing users to generate digits, save them, and reconstruct handwritten numbers from stored samples.

---

## Dataset

* **Dataset:** MNIST (handwritten digits)
* **Total images:** 70,000
* **Training set:** 60,000 images
* **Test set:** 10,000 images
* **Image size:** 28 × 28 pixels (grayscale)
* **Number of classes:** 10 (digits 0–9)
* **Pixel range:** 0–255 (normalized to [-1, 1])

---

## Preprocessing

* Loaded using `tf.keras.datasets.mnist.load_data()`
* Reshaped to **28×28×1** for CNN-based GAN input
* Normalized pixel values to [-1, 1] using:

  ```
  x_norm = (x - 127.5) / 127.5
  ```
* Data batched and shuffled using `tf.data.Dataset`

---

## Model Architecture

### Generator

* Input: Random noise vector (100 dimensions)
* Dense + BatchNorm + LeakyReLU
* Reshape to 7×7×256
* Conv2DTranspose layers
* Output: 28×28×1 image (tanh activation)

### Discriminator

* Conv2D layers with LeakyReLU
* Dropout for regularization
* Output: Binary classification (real/fake)

---

## Training

* Loss: Binary Crossentropy
* Optimizer: Adam (lr = 0.0001)
* Training mode: adversarial (Generator vs Discriminator)
* Output images generated every epoch for monitoring

---

## Web Application

### Backend (Flask)

Provides APIs:

* `/generate` → Generate new digit image
* `/save` → Save generated image under a digit label
* `/get/<digit>` → Retrieve saved digit image
* `/write_number/<number>` → Generate full handwritten number from stored digits

### Frontend

* HTML + JavaScript + Canvas
* Features:

  * Generate digit
  * Save digit to database
  * Load digit from database
  * Write full number using saved digits

---

## Feature: Handwritten Number Generation

The system can reconstruct a full number by stitching saved digit images horizontally.

### Process

1. User enters number (e.g., 123)
2. Backend fetches stored images for each digit
3. Images are concatenated using NumPy `hstack`
4. Combined image is returned to frontend

---

## Limitations

* Uses in-memory storage (not persistent database)
* Standard GAN generates random digits only (not controllable)
* Number generation depends on saved digit availability

---

## Future Improvements

* Replace GAN with Conditional GAN (cGAN)
* Add persistent database (SQLite / MongoDB)
* Improve handwriting realism with sequence models
* Add downloadable image export

---

## Tech Stack

* Python
* TensorFlow / Keras
* Flask
* JavaScript
* HTML Canvas

---

## Author Notes

This project demonstrates generative modeling and web integration for AI-based image synthesis and interaction.
