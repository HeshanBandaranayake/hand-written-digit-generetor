import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import os
import sys

def build_generator(latent_dim, num_classes=10):
    # Label input
    label_input = Input(shape=(1,), name='label_input')
    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    
    # Noise input
    noise_input = Input(shape=(latent_dim,), name='noise_input')
    
    # Combine noise and label
    combined_input = layers.Concatenate()([noise_input, label_embedding])
    
    # Foundation for 7x7 image
    x = layers.Dense(256 * 7 * 7)(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((7, 7, 256))(x)
    
    # Upsample to 14x14
    x = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Upsample to 28x28
    x = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    output_img = layers.Conv2D(1, (7,7), activation='tanh', padding='same')(x)
    
    return models.Model([noise_input, label_input], output_img, name='generator')

def build_discriminator(img_shape=(28, 28, 1), num_classes=10):
    # Image input
    img_input = Input(shape=img_shape, name='img_input')
    
    # Label input
    label_input = Input(shape=(1,), name='label_input')
    label_embedding = layers.Embedding(num_classes, np.prod(img_shape))(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
    
    # Concatenate image and label
    combined_input = layers.Concatenate()([img_input, label_embedding])
    
    # Downsample to 14x14
    x = layers.Conv2D(64, (3,3), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Downsample to 7x7
    x = layers.Conv2D(64, (3,3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    
    # Classifier
    x = layers.Flatten()(x)
    output_validity = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model([img_input, label_input], output_validity, name='discriminator')

def train_gan():
    latent_dim = 100
    num_classes = 10
    steps = 3000  # More steps for better separation
    batch_size = 64
    half_batch = int(batch_size / 2)
    
    print("Loading MNIST dataset...")
    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    
    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.0
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = y_train.reshape(-1, 1)
    
    # Build models
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5) # Slightly slower G
    
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    
    generator = build_generator(latent_dim)
    
    # Combined model
    discriminator.trainable = False
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    img = generator([noise, label])
    validity = discriminator([img, label])
    
    combined = models.Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    
    d_losses = []
    g_losses = []
    
    print(f"Starting cGAN training for {steps} steps...")
    for step in range(steps):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Select random real images and labels
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]
        real_labels = y_train[idx]
        
        # Generate fake images and labels
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_labels = np.random.randint(0, 10, (half_batch, 1))
        gen_imgs = generator.predict([noise, fake_labels], verbose=0)
        
        # Soft labels
        y_real = np.ones((half_batch, 1)) * 0.9
        y_fake = np.zeros((half_batch, 1))
        
        d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], y_real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, fake_labels], y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))
        y_gen = np.ones((batch_size, 1))
        
        g_loss = combined.train_on_batch([noise, sampled_labels], y_gen)
        
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        if step % 50 == 0:
            print(f"Step {step}/{steps} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            sys.stdout.flush()

    print("Training finished. Generating samples...")
    
    # Save loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.title("cGAN Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gan_loss_plot.png')
    plt.close()
    
    # Generate samples (rows: digits 0-9)
    r, c = 10, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    sampled_labels = np.array([[i] * c for i in range(r)]).flatten().reshape(-1, 1)
    gen_imgs = generator.predict([noise, sampled_labels], verbose=0)
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r, c, figsize=(5, 10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig('generated_digits.png')
    plt.close()
    
    generator.save('generator.keras')
    print("Saved generator.keras and sample images.")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_gan()
