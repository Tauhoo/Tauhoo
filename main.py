from src.configure import configure
from src.image_reader import image_reader
import tensorflow as tf
from tensorflow import keras


""" initialize config file """
config = configure()
learning_rate = config.setting['learning_rate']
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']
image_size = int(config.setting['image_size'])

""" initialize dataset """
reader = image_reader(data_folder_path, image_size).shuffle()
generator = reader.get_generator()
pokemon_list = reader.pokemon_list
pokemon_size = len(pokemon_list)

""" create model """
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(image_size, image_size)))
model.add(keras.layers.Dense(pokemon_size, activation=tf.nn.softmax))
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    metrics=['accuracy'])
model.summary()

""" train """
model.fit(generator, epochs=6, steps_per_epoch=100)

""" save weight """
model.save(weight_file_path)
