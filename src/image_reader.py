import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import sys


class image_reader:
    data_list = []

    def __init__(self, path, image_size):
        self.root_path = path
        self.pokemon_list = os.listdir(path)
        self.image_size = image_size

        print("> load dataset from folder")

        for index, pokemon in enumerate(self.pokemon_list):
            folder_path = path + "/" + pokemon
            file_names = os.listdir(folder_path)

            for file_name in file_names:
                file_path = folder_path + "/" + file_name
                self.data_list.append((index, file_path))

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def get_generator(self):
        for label, path in self.data_list:
            try:
                image = tf.io.read_file(path)
                image = tf.image.decode_image(image, channels=1)
                image = tf.image.resize(
                    image, [self.image_size, self.image_size])
                yield image/255, label, [None]
            except:
                print(path)

    def show_image(self, image):
        image = np.array(image).reshape((self.image_size, self.image_size))
        plt.imshow(image)
        plt.show()
        return self
