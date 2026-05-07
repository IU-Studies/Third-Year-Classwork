"""
Develop an autoencoder to encode and decode the image. Analyse the results. 
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input,
    Dense
)

from tensorflow.keras.models import Model

from tensorflow.keras.datasets import mnist

# ==============================
# LOAD MNIST DATASET
# ==============================

(x_train, _), (x_test, _) = mnist.load_data()

# ==============================
# NORMALIZE DATA
# ==============================

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ==============================
# RESHAPE TO VECTORS
# ==============================

x_train = x_train.reshape((len(x_train), 28 * 28))
x_test = x_test.reshape((len(x_test), 28 * 28))

# ==============================
# MODEL PARAMETERS
# ==============================

input_dim = 784
encoding_dim = 64   # Compressed representation size

# ==============================
# ENCODER
# ==============================

input_img = Input(shape=(input_dim,))

encoded = Dense(
    128,
    activation='relu'
)(input_img)

encoded = Dense(
    encoding_dim,
    activation='relu'
)(encoded)

# ==============================
# DECODER
# ==============================

decoded = Dense(
    128,
    activation='relu'
)(encoded)

decoded = Dense(
    input_dim,
    activation='sigmoid'
)(decoded)

# ==============================
# AUTOENCODER MODEL
# ==============================

autoencoder = Model(
    input_img,
    decoded
)

# ==============================
# ENCODER MODEL
# (Compressed Representation)
# ==============================

encoder = Model(
    input_img,
    encoded
)

# ==============================
# COMPILE MODEL
# ==============================

autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# ==============================
# TRAIN MODEL
# ==============================

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# ==============================
# ENCODE & RECONSTRUCT IMAGES
# ==============================

encoded_imgs = encoder.predict(x_test)

decoded_imgs = autoencoder.predict(x_test)

# ==============================
# DISPLAY RESULTS
# ==============================

n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # --------------------------
    # ORIGINAL IMAGE
    # --------------------------

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(
        x_test[i].reshape(28, 28)
    )

    plt.title("Original")

    plt.gray()

    plt.axis("off")

    # --------------------------
    # RECONSTRUCTED IMAGE
    # --------------------------

    ax = plt.subplot(2, n, i + n + 1)

    plt.imshow(
        decoded_imgs[i].reshape(28, 28)
    )

    plt.title("Reconstructed")

    plt.gray()

    plt.axis("off")

plt.show()
