import tensorflow as tf
import numpy as np
from CNN_ImageModel import ImageModel

############ EXAMPLE OF MNIST INSTANCE ############

#load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#generate model instance
model = ImageModel(x_train.shape[1], x_train.shape[2], x_train.shape[3], 10, "mnist model")
model.compile_model(1e-4)

#train and test
model.train_model(x_train, y_train, 200, 1000)

y_predict = model.test_model(x_test)
ImageModel.getAccuracy(y_predict, y_test)