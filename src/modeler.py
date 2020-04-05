from src.configure import configure
from os import path
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from src.image_reader import image_reader
import numpy as np


class modeler:
    def __init__(self):
        """ initialize config file """
        config = configure()
        self.learning_rate = config.setting['learning_rate']
        self.data_folder_path = config.source['data_folder_path']
        self.weight_file_path = config.source['weight_file_path']
        self.image_size = int(config.setting['image_size'])
        self.epochs = int(config.setting['epochs'])
        self.steps_per_epoch = int(config.setting['steps_per_epoch'])
        self.initial_generator()

    def initial_generator(self):
        """ initialize dataset """
        self.reader = image_reader(
            self.data_folder_path, self.image_size).shuffle()
        self.generator = self.reader.get_generator()
        self.pokemon_list = self.reader.pokemon_list
        self.pokemon_size = len(self.pokemon_list)
        return self

    def create_model(self, summary=False):
        """ create model """
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(20, (10, 10), activation='relu',
                                           input_shape=(self.image_size, self.image_size, 3)))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(self.pokemon_size))
        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy'])
        if summary:
            self.model.summary()
        return self

    def load_weight(self):
        """ load weights """
        if path.exists(self.weight_file_path):
            print("already have weight {}".format(self.weight_file_path))
            self.model.load_weights(self.weight_file_path)
        else:
            print("not found {}".format(self.weight_file_path))
        return self

    def train(self, save_weight=True):
        """ train """
        self.history = self.model.fit_generator(
            self.generator, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)
        if save_weight:
            """ save weight """
            self.model.save_weights(self.weight_file_path)
            print('weights were saved.')
        return self

    def predict(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(
            image, [self.image_size, self.image_size])
        image /= 255
        x = tf.expand_dims(image, 0)
        predictions = self.model.predict(x)

        return self.reader.pokemon_list[np.argmax(predictions)]

    def plot_train_history(self):
        """ graph """
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()
        return self
