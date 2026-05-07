"""
Develop face recognition system using CNN. Create a dataset of minimum 50 
students from your class. Check and validate the accuracy of the model. 
Apply dimensionality reduction on input image and plot the change in 
accuracy of system.
"""

# ==============================
# FACE RECOGNITION USING CNN + PCA (Exp 4)
# ==============================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people

IMG_SIZE = 100

print("\nLoading LFW dataset...")

lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)

X = lfw.images
y = lfw.target
class_names = lfw.target_names

print("Original shape:", X.shape)

# Convert grayscale → RGB
X = np.array([
    cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for img in X
])

# Resize images
X = np.array([
    cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    for img in X
])

# Normalize
X = X / 255.0

print("Processed shape:", X.shape)
print("Number of classes:", len(class_names))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# CNN MODEL
# ==============================

print("\nTraining CNN model...")

model = models.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(100, 100, 3)
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(
        len(class_names),
        activation='softmax'
    )
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Evaluate CNN
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\nCNN Accuracy:", test_acc)

# ==============================
# PCA + SVM
# ==============================

print("\nApplying PCA...")

# Flatten images
X_flat = X.reshape(len(X), -1)

dims = [50, 100, 200, 300]
accuracies = []

for d in dims:

    print(f"\nProcessing PCA with {d} dimensions...")

    pca = PCA(n_components=d)

    X_pca = pca.fit_transform(X_flat)

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42
    )

    clf = SVC()

    clf.fit(X_train_pca, y_train_pca)

    acc = clf.score(X_test_pca, y_test_pca)

    accuracies.append(acc)

    print(f"Accuracy with {d} dimensions: {acc}")

# ==============================
# GRAPH
# ==============================

plt.plot(dims, accuracies, marker='o')

plt.xlabel("Number of Dimensions")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Dimensionality Reduction")

plt.grid()

plt.show()

# ==============================
# FINAL RESULTS
# ==============================

print("\n===== FINAL RESULTS =====")

print("CNN Accuracy:", test_acc)

for d, acc in zip(dims, accuracies):
    print(f"PCA ({d} dims) Accuracy: {acc}")
