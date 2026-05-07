"""
Write a program to demonstrate the change in accuracy/loss/convergence time with 
change in optimizers like stochastic gradient descent, adam, adagrad, RMSprop and 
Nadam for any suitable application
"""


import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt

# ==============================
# LOAD DATASET (Handwritten Digits)
# ==============================

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# ==============================
# FUNCTION TO CREATE MODEL
# ==============================

def create_model():

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),   # FIXED warning

        keras.layers.Flatten(),

        keras.layers.Dense(
            128,
            activation='relu'
        ),

        keras.layers.Dense(
            10,
            activation='softmax'
        )
    ])

    return model

# ==============================
# LIST OF OPTIMIZERS
# ==============================

optimizers = {
    "SGD": keras.optimizers.SGD(),
    "Adam": keras.optimizers.Adam(),
    "Adagrad": keras.optimizers.Adagrad(),
    "RMSprop": keras.optimizers.RMSprop(),
    "Nadam": keras.optimizers.Nadam()
}

results = {}

# ==============================
# TRAIN MODEL WITH EACH OPTIMIZER
# ==============================

for name, opt in optimizers.items():

    print(f"\nTraining with {name} optimizer...")

    model = create_model()

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=0
    )

    end_time = time.time()

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    results[name] = {
        "accuracy": accuracy,
        "loss": loss,
        "time": end_time - start_time,
        "history": history
    }

# ==============================
# PRINT RESULTS
# ==============================

print("\nFinal Results:\n")

for name, res in results.items():

    print(
        f"{name}: "
        f"Accuracy={res['accuracy']:.4f}, "
        f"Loss={res['loss']:.4f}, "
        f"Time={res['time']:.2f}s"
    )

# ==============================
# PLOT ACCURACY GRAPH
# ==============================

plt.figure()

for name in results:
    plt.plot(
        results[name]["history"].history['val_accuracy'],
        label=name
    )

plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")

plt.legend()

plt.show()

# ==============================
# PLOT LOSS GRAPH
# ==============================

plt.figure()

for name in results:
    plt.plot(
        results[name]["history"].history['val_loss'],
        label=name
    )

plt.title("Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")

plt.legend()

plt.show()
