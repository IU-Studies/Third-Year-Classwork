"""
Apply transfer learning with pre-trained VGG16/ResNet50/MobileNet model on 
given dataset and analyze the results
"""
import tensorflow as tf

from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    MobileNet
)

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D
)

from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

# ==============================
# LOAD CIFAR-10 DATASET
# ==============================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

NUM_CLASSES = 10
BATCH_SIZE = 32

# ==============================
# PREPROCESS FUNCTION
# ==============================

def preprocess(image, label):

    # Resize images
    image = tf.image.resize(image, (224, 224))

    # Normalize images
    image = image / 255.0

    # One-hot encode labels
    label = tf.one_hot(label[0], NUM_CLASSES)

    return image, label

# ==============================
# CREATE DATASETS
# ==============================

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

train_ds = train_ds.map(preprocess).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)
)

test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)

# ==============================
# BUILD MODEL FUNCTION
# ==============================

def build_model(base_model):

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
        activation='relu'
    )(x)

    output = Dense(
        NUM_CLASSES,
        activation='softmax'
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ==============================
# LOAD PRETRAINED MODELS
# ==============================

models = {
    "VGG16": VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    ),

    "ResNet50": ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    ),

    "MobileNet": MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
}

EPOCHS = 3   # Reduced for speed

history_dict = {}

# ==============================
# TRAIN EACH MODEL
# ==============================

for name, base in models.items():

    print(f"\nTraining {name}...")

    model = build_model(base)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS
    )

    history_dict[name] = history.history

# ==============================
# PLOT VALIDATION ACCURACY
# ==============================

for name, history in history_dict.items():

    plt.plot(
        history['val_accuracy'],
        label=name
    )

plt.legend()

plt.title("Model Comparison")

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")

plt.show()
