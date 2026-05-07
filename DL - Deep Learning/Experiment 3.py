"""
Develop classification model for cat-dogs dataset using cnn model.
Analyze the model accuracy and generate classification report.
● develop an GUI and test the user given inputs.
● Analyze the result with and without regularization/dropout
"""

# =====================================================
# CAT vs DOG CLASSIFICATION USING CNN
# =====================================================

# Install libraries
!pip install tensorflow scikit-learn seaborn -q

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

print("Libraries Loaded")

# =====================================================
# LOAD DATASET (TensorFlow built-in CIFAR-10)
# =====================================================

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 classes
# Cat = 3
# Dog = 5

cat_class = 3
dog_class = 5

# Filter only cats and dogs
train_filter = (train_labels == cat_class) | (train_labels == dog_class)
test_filter = (test_labels == cat_class) | (test_labels == dog_class)

train_images = train_images[train_filter.flatten()]
train_labels = train_labels[train_filter.flatten()]

test_images = test_images[test_filter.flatten()]
test_labels = test_labels[test_filter.flatten()]

# Convert labels
# Cat = 0
# Dog = 1

train_labels = (train_labels == dog_class).astype(int)
test_labels = (test_labels == dog_class).astype(int)

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

print("Dataset Ready")

# =====================================================
# CNN MODEL WITHOUT REGULARIZATION
# =====================================================

print("\nTraining CNN WITHOUT Regularization")

model = models.Sequential([
    
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(32, 32, 3)
    ),
    
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dense(
        1,
        activation='sigmoid'
    )
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

loss, acc = model.evaluate(test_images, test_labels)

print("\nAccuracy WITHOUT Regularization:", acc)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

pred = model.predict(test_images)
pred = (pred > 0.5).astype(int)

print("\nClassification Report\n")

print(classification_report(test_labels, pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(test_labels, pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Without Regularization")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# CNN MODEL WITH DROPOUT REGULARIZATION
# =====================================================

print("\nTraining CNN WITH Dropout Regularization")

model_reg = models.Sequential([

    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(32, 32, 3)
    ),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation='sigmoid'
    )
])

model_reg.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_reg = model_reg.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

loss_reg, acc_reg = model_reg.evaluate(test_images, test_labels)

print("\nAccuracy WITH Dropout:", acc_reg)

# =====================================================
# ACCURACY COMPARISON GRAPH
# =====================================================

plt.figure(figsize=(6, 4))

plt.plot(history.history['accuracy'])
plt.plot(history_reg.history['accuracy'])

plt.legend([
    "Without Regularization",
    "With Dropout"
])

plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()

print("\nExperiment Completed Successfully")
