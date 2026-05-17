import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize Data
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Evaluate Model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", accuracy)

# Predict Sample Digit
prediction = model.predict(
    np.expand_dims(X_test[0], axis=0)
)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)

# Display Digit Image
plt.imshow(X_test[0], cmap='gray')

plt.title(f"Predicted Digit: {predicted_digit}")

plt.show()
