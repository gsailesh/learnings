import tensorflow as tf
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

# Read and standardize the data.
boston_housing_data = tf.keras.datasets.boston_housing
(raw_X_train, y_train), (raw_X_test, y_test) = boston_housing_data.load_data()

X_mean = np.mean(raw_X_train, axis=0)
X_sd = np.std(raw_X_train, axis=0)

X_train = (raw_X_train - X_mean) / X_sd
X_test = (raw_X_test - X_mean) / X_sd

# Create and train model
model_v1 = tf.keras.models.Sequential()

model_v1.add(tf.keras.layers.Dense(64, activation="relu", input_shape=[13]))
model_v1.add(tf.keras.layers.Dense(64, activation="relu"))
model_v1.add(tf.keras.layers.Dense(1, activation="linear"))

model_v1.compile(
    loss="mean_squared_error",
    optimizer="Adam",
    metrics=["mean_absolute_error"]
)

model_v1.summary()

# hist_v1 = model_v1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
#                        validation_data=(X_test, y_test),
#                        shuffle=True)
'''
Epoch 500/500
26/26 [==============================] - 0s 5ms/step - loss: 0.7129 - mean_absolute_error: 0.5750 - val_loss: 15.5159 - val_mean_absolute_error: 2.6775
'''
# Predictions
predictions = model_v1.predict(X_test)
for i in range(0, 4):
    print(predictions[i], y_test[i])

# V2 with regularization

model_v2 = tf.keras.models.Sequential()

model_v2.add(
    tf.keras.layers.Dense(64, activation="relu", input_shape=[13], kernel_regularizer=tf.keras.regularizers.l2(0.1),
                          bias_regularizer=tf.keras.regularizers.l2(0.1)))
model_v2.add(tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                   bias_regularizer=tf.keras.regularizers.l2(0.1)))
model_v2.add(tf.keras.layers.Dense(1, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                   bias_regularizer=tf.keras.regularizers.l2(0.1)))

model_v2.compile(
    loss="mean_squared_error",
    optimizer="Adam",
    metrics=["mean_absolute_error"]
)

model_v2.summary()

# hist_v2 = model_v2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
#                        validation_data=(X_test, y_test),
#                        shuffle=True)
'''
Epoch 500/500
26/26 [==============================] - 0s 6ms/step - loss: 9.0916 - mean_absolute_error: 1.5963 - val_loss: 20.6900 - val_mean_absolute_error: 2.6542
'''
# Predictions
predictions = model_v2.predict(X_test)
for i in range(0, 4):
    print(predictions[i], y_test[i])

# V3 - Dropout

model_v3 = tf.keras.models.Sequential()

model_v3.add(
    tf.keras.layers.Dense(64, activation="relu", input_shape=[13]))
model_v3.add(tf.keras.layers.Dropout(0.2))
model_v3.add(tf.keras.layers.Dense(64, activation="relu"))
model_v3.add(tf.keras.layers.Dropout(0.2))
model_v3.add(tf.keras.layers.Dense(1, activation="linear"))

model_v3.compile(
    loss="mean_squared_error",
    optimizer="Adam",
    metrics=["mean_absolute_error"]
)

model_v3.summary()

hist_v3 = model_v3.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                       validation_data=(X_test, y_test),
                       shuffle=True)
'''
Epoch 500/500
26/26 [==============================] - 0s 6ms/step - loss: 2.1127 - mean_absolute_error: 1.0729 - val_loss: 14.2601 - val_mean_absolute_error: 2.6291
'''

# Predictions
predictions = model_v3.predict(X_test)
for i in range(0, 4):
    print(predictions[i], y_test[i])

'''
TO DO:
OPTIMAL CONFIG:

Conf6 -> 128/128/64/1 -> Dropout=0.3 -> 2.38 -> 2.31
'''
