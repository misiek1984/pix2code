__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

from .WeightsSaver import *

class AModel:
    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""

    def configure_callbacks(self):
        #filepath="{}/{}-epoch-{{epoch:02d}}-loss-{{val_loss:.2f}}.h5".format(self.output_path, self.name)
        #modelCheckpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

        weightsSaver = WeightsSaver(self.output_path, self.name, self.model, 100)
        return [weightsSaver]

    def save(self, only_model=False):
        model_json = self.model.to_json()
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)
        if only_model == False:
            self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))
