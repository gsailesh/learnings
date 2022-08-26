import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 150
BATCH_SIZE = 32

cifar10 = tf.keras.datasets.cifar10
NUM_CLS = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

mean = np.mean(X_train)
sd = np.std(X_train)

X_train_norm = (X_train - mean) / sd
X_test_norm = (X_test - mean) / sd

y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLS)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLS)

# Model arch, summary
model_v1 = tf.keras.models.Sequential()

model_v1.add(
    tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), activation="relu", padding="same", input_shape=(32, 32, 3),
                           kernel_initializer="he_normal", bias_initializer="zeros"))
model_v1.add(
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3),
                           kernel_initializer="he_normal", bias_initializer="zeros"))
model_v1.add(tf.keras.layers.Flatten())
model_v1.add(
    tf.keras.layers.Dense(10, activation="softmax", kernel_initializer="glorot_uniform", bias_initializer="zeros"))

model_v1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model_v1.summary())

# Fit the model
hist_v1 = model_v1.fit(X_train_norm, y_train_ohe, validation_data=(X_test_norm, y_test_ohe), epochs=EPOCHS,
                       batch_size=BATCH_SIZE, verbose=2, shuffle=True)

print(hist_v1.history)
train_error = hist_v1.history["loss"]
val_error = hist_v1.history["val_loss"]
epochs = list(range(0, EPOCHS))

_, ax = plt.subplots(figsize=(10, 8))
plt.plot(epochs, train_error)
plt.plot(epochs, val_error)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation loss over epochs (Cifar10)")

plt.show()
