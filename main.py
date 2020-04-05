from src.configure import configure
from src.image_reader import image_reader
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from os import path


""" initialize config file """
config = configure()
learning_rate = config.setting['learning_rate']
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = int(config.setting['image_size'])
epochs = int(config.setting['epochs'])
steps_per_epoch = int(config.setting['steps_per_epoch'])

""" initialize dataset """
reader = image_reader(data_folder_path, image_size).shuffle()
generator = reader.get_generator()
pokemon_list = reader.pokemon_list
pokemon_size = len(pokemon_list)


""" create model """
model = keras.Sequential()
model.add(keras.layers.Conv2D(20, (10, 10), activation='relu',
                              input_shape=(image_size, image_size, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(pokemon_size))
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    metrics=['accuracy'])
model.summary()

""" load weights """
if path.exists(weight_file_path):
    print("already have weight {}".format(weight_file_path))
    model.load_weights(weight_file_path)

""" train """
history = model.fit_generator(
    generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

""" save weight """
model.save_weights(weight_file_path)

""" graph """
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
