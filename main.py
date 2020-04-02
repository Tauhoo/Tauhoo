from src.configure import configure
from src.image_reader import image_reader

config = configure()
learning_rate = config.setting['learning_rate']
data_folder_path = config.source['data_folder_path']
weight_file_path = config.source['weight_file_path']

reader = image_reader(data_folder_path)
