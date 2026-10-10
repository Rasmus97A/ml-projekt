import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

digits = load_digits()
X = digits.images / 16.0          # scale pixel values from 0-16 down to 0-1
X = X[..., np.newaxis]            # (1797, 8, 8) -> (1797, 8, 8, 1): add a "colour channel"
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape)

model = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_split=0.15, epochs=30, batch_size=32, verbose=0)

pred = np.argmax(model.predict(X_test), axis=1)
print("CNN accuracy:", f"{accuracy_score(y_test, pred):.2%}")

# Training curves
plt.plot(history.history["accuracy"], label="training")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("CNN learning curve")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Purples")
plt.title("CNN on digits")
plt.show()

wrong = np.where(pred != y_test)[0]       # positions where the guess was wrong
print("Number of mistakes:", len(wrong))

fig, axes = plt.subplots(1, len(wrong), figsize=(2 * len(wrong), 2.5))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(X_test[i].squeeze(), cmap="gray")
    ax.set_title(f"true {y_test[i]}\nguess {pred[i]}")
    ax.axis("off")
plt.show()