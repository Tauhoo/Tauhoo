# Pokemon classifier

Pokemon classifier is a Python project that implements machine learning for classifing pokemon by its picture.

## Installation

Use the package manager [pip](https://help.dreamhost.com/hc/en-us/articles/115000702772-Installing-a-custom-version-of-Python-3) to install.

```bash
pip install tensorflow
pip install matplotlib
pip install numpy
```

And you need to put your data that you want to train.

```
.
├── files
|   └── data
|     ├── pokemon-name1 #pokemon name
|     |   ├── pic1.png # pokemon's picture that will be trained.
|     |   ├── pic2.png
|     |   └── ...
|     ├── pokemon-name2
|     |   ├── pic1.png
|     |   ├── pic2.png
|     |   └──...
|     └── ...
├── src
├── config.ini # config training
├── test.py
└── train.py
```

## Config

```
[source]
   data_folder_path = ./files/data # Your data path. you can change it.
   weight_file_path = ./files/weight.h5 # your weight file path. it will be load and update.
[setting]
   image_size = 80 # Size of your image that will be resized before trained.
   epochs = 10
   steps_per_epoch = 100
```

- if you don't have the weight file, you just put the target path that you want to save your weight in weight_file_path.

## Usage

Config your training.
Train by run this command.

```bash
python3.7 train.py
```

Test with CLI.

```bash
python3.7 test.py
```
