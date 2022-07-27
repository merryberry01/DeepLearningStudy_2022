import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris['data'],iris['target'], test_size = 0.25)
y_train_enc = tf.one_hot(y_train, 3, on_value = 1, off_value = 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (4,)))
model.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))

sgd = tf.optimizers.SGD(learning_rate = 1e-2)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

history = model.fit(x_train, y_train_enc, epochs = 50, batch_size = 1, shuffle = False)
y_pred = model.predict(x_test)

plt.scatter(range(0, len(y_test)), tf.math.argmax(y_pred, axis = 1) - y_test)
plt.show()