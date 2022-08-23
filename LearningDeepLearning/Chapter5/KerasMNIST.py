import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(42)

EPOCHS = 10
BATCH_SIZE = 1

# Load the training and test data
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Standadrize the data
mean = np.mean(X_train);
sd = np.std(X_train)

X_train_std = X_train - mean / sd
X_test_std = X_test - mean / sd

y_train_ohe = to_categorical(y_train, num_classes=10)
y_test_ohe = to_categorical(y_test, num_classes=10)

# Initializing the weights
# init = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

# # Model architecture
# model = tf.keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(25, activation='tanh', kernel_initializer=init, bias_initializer='zeros'),
#     keras.layers.Dense(10, activation='sigmoid', kernel_initializer=init, bias_initializer='zeros')
# ])
#
# # Compile the model
# model.compile(
#     loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["accuracy"]
# )
#
# # Fit the model
# hist = model.fit(X_train_std, y_train_ohe, validation_data=(X_test_std, y_test_ohe), epochs=EPOCHS,
#                  batch_size=BATCH_SIZE,
#                  verbose=2, shuffle=True)

### Config 5
# Model architecture
model_v2 = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(25, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'),
    keras.layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros')
])

# Compile the model
model_v2.compile(
    loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"]
)

# Fit the model
hist_v2 = model_v2.fit(X_train_std, y_train_ohe, validation_data=(X_test_std, y_test_ohe), epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       verbose=1, shuffle=True)
