import tensorflow as tf
import numpy as np
from CNN_ImageModel import ImageModel

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

models = []
model_num = 7
for i in range(model_num):
    models.append(ImageModel(x_train.shape[1], x_train.shape[2], x_train.shape[3], 10, "Model " + str(i + 1)))

#Training
for model in models:
    model.compile_model(1e-4)
    print(f"{model.name} started Learning")
    model.train_model(x_train, y_train, 1, 1000)

#Testing
y_predict = np.full((y_test.shape[0], 10), 0.)
for model in models:
    y_predict += model.test_model(x_test)

ImageModel.getAccuracy(y_predict, y_test)