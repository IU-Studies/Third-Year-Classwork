"""
Case study GAN
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
    Input
)

from tensorflow.keras.optimizers import Adam

# ==============================
# LOAD MNIST DATASET
# ==============================

(X_train, _), (_, _) = mnist.load_data()

# Normalize images to [-1, 1]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5

# Add channel dimension
X_train = np.expand_dims(X_train, axis=-1)

# ==============================
# PARAMETERS
# ==============================

img_shape = (28, 28, 1)
latent_dim = 100

# ==============================
# BUILD GENERATOR
# ==============================

def build_generator():

    model = Sequential([

        Input(shape=(latent_dim,)),

        Dense(256),
        LeakyReLU(0.2),

        Dense(512),
        LeakyReLU(0.2),

        Dense(1024),
        LeakyReLU(0.2),

        Dense(
            28 * 28,
            activation='tanh'
        ),

        Reshape(img_shape)
    ])

    return model

# ==============================
# BUILD DISCRIMINATOR
# ==============================

def build_discriminator():

    model = Sequential([

        Input(shape=img_shape),

        Flatten(),

        Dense(512),
        LeakyReLU(0.2),

        Dense(256),
        LeakyReLU(0.2),

        Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.0002, 0.5),
        metrics=['accuracy']
    )

    return model

# ==============================
# CREATE MODELS
# ==============================

generator = build_generator()

discriminator = build_discriminator()

# Freeze discriminator while training GAN
discriminator.trainable = False

# ==============================
# BUILD GAN
# ==============================

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5)
)

# ==============================
# TRAIN FUNCTION
# ==============================

def train(epochs, batch_size=64, sample_interval=500):

    # Labels
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # --------------------------
        # TRAIN DISCRIMINATOR
        # --------------------------

        idx = np.random.randint(
            0,
            X_train.shape[0],
            batch_size
        )

        real_imgs = X_train[idx]

        noise = np.random.normal(
            0,
            1,
            (batch_size, latent_dim)
        )

        gen_imgs = generator.predict(
            noise,
            verbose=0
        )

        d_loss_real = discriminator.train_on_batch(
            real_imgs,
            valid
        )

        d_loss_fake = discriminator.train_on_batch(
            gen_imgs,
            fake
        )

        d_loss = 0.5 * np.add(
            d_loss_real,
            d_loss_fake
        )

        # --------------------------
        # TRAIN GENERATOR
        # --------------------------

        noise = np.random.normal(
            0,
            1,
            (batch_size, latent_dim)
        )

        g_loss = gan.train_on_batch(
            noise,
            valid
        )

        # --------------------------
        # PRINT PROGRESS
        # --------------------------

        if epoch % sample_interval == 0:

            print(
                f"{epoch} "
                f"[D loss: {d_loss[0]:.4f}, "
                f"acc: {100 * d_loss[1]:.2f}%] "
                f"[G loss: {g_loss:.4f}]"
            )

            sample_images(generator, epoch)

# ==============================
# GENERATE SAMPLE IMAGES
# ==============================

def sample_images(generator, epoch):

    noise = np.random.normal(
        0,
        1,
        (10, latent_dim)
    )

    gen_imgs = generator.predict(
        noise,
        verbose=0
    )

    # Rescale images to [0,1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    plt.figure(figsize=(10, 2))

    for i in range(10):

        plt.subplot(1, 10, i + 1)

        plt.imshow(
            gen_imgs[i, :, :, 0],
            cmap='gray'
        )

        plt.axis('off')

    plt.show()

# ==============================
# TRAIN GAN
# ==============================

train(
    epochs=5000,
    batch_size=64,
    sample_interval=500
)
