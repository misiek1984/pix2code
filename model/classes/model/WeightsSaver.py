from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, output_path, name, model, N):
        self.output_path = output_path
        self.name = name
        self.model = model
        self.N = N
        self.batch = 0
        self.epoch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            self.save(logs)
        self.batch += 1

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        self.save(logs)
        self.batch = 0

    def save(self, logs={}):
        filepath = "{}/{}-epoch-{:02d}-batch-{:02d}-loss-{:.2f}-only-weights.h5".format(self.output_path, self.name, self.epoch,
                                                                           self.batch, logs.get('loss'))
        self.model.save_weights(filepath)

        filepath = "{}/{}-epoch-{:02d}-batch-{:02d}-loss-{:.2f}-full-model.h5".format(self.output_path, self.name,
                                                                                        self.epoch,
                                                                                        self.batch, logs.get('loss'))
        self.model.save(filepath)

        model_json = self.model.to_json()
        filepath = "{}/{}-epoch-{:02d}-batch-{:02d}-loss-{:.2f}-only-model.json".format(self.output_path, self.name,
                                                                                      self.epoch,
                                                                                      self.batch, logs.get('loss'))
        with open(filepath, "w") as json_file:
            json_file.write(model_json)