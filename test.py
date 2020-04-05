from src.modeler import modeler
from os import path

model = modeler().create_model().load_weight()

while True:
    image_path = input("Image's path (type q or nothing to exit): ").strip()
    if image_path == 'q' or image_path == '':
        print('Goodbye.')
        break
    if path.exists(image_path):
        result = model.predict(image_path)
        print("Pokemon in this picture is {}".format(result))
    else:
        print("can't find {}".format(image_path))
