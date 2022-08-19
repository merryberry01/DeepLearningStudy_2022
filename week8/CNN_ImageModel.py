import tensorflow as tf
import numpy as np

class ImageModel:
    def __init__(self, width, height, color, output_cnt, name):
        self.name = name
        self.outputCnt = output_cnt

        self.width = width
        self.height = height
        self.color = color
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape = (width, height, color)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.build_convNet()
        self.build_fcNet()

    def build_convNet(self):
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(self.width, self.height, self.color)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')) #size = (width - 2) / 2 + 1 = width/2
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(self.width/2, self.height/2, 32)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(self.width/2, self.height/2, 32)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(self.width/2, self.height/2, 32)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
        self.model.add(tf.keras.layers.Dropout(0.2))

    def build_fcNet(self):
        input_sz = self.model.output_shape[1] * self.model.output_shape[2] * self.model.output_shape[3]
        self.model.add(tf.keras.layers.Reshape((input_sz, )))
        
        xavier_init = tf.keras.initializers.GlorotUniform() #GlorotUniform(), GlorotNormal()
        he_init = tf.keras.initializers.HeUniform() #HeUniform(), HeNormal()

        self.model.add(tf.keras.layers.Dense(units = 50, input_shape = (input_sz, ), activation = 'relu', kernel_initializer = he_init))
        self.model.add(tf.keras.layers.Dense(units = 50, input_shape = (50, ), activation = 'relu', kernel_initializer = he_init))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units = 50, input_shape = (50, ), activation = 'relu', kernel_initializer = he_init))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(units = self.outputCnt, input_shape = (50, ), activation = 'softmax', kernel_initializer = xavier_init))

    def compile_model(self, lr):
        adam = tf.optimizers.Adam(learning_rate = lr)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = adam)
    
    def train_model(self, xTrain, yTrain, epoch, batch_sz):
        yTrain_enc = tf.one_hot(yTrain, self.outputCnt, on_value = 1, off_value = 0)
        self.model.fit(xTrain, yTrain_enc, epochs = epoch, batch_size = batch_sz)

    def test_model(self, xTest):
        return self.model.predict(xTest)

    def getAccuracy(y_pred, y_test):
        res = np.array(tf.math.argmax(y_pred, axis = 1) - y_test)
        correct = (res == 0).sum()
        incorrect = (res != 0).sum()
        print(f"correct: {correct}, incorrect: {incorrect}, accuracy: {(correct / res.size) * 100}%")