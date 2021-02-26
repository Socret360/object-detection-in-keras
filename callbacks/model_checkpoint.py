import os
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(self, output_dir, epoch_frequency, iteration_frequency):
        self.output_dir = output_dir
        self.iteration_frequency = iteration_frequency
        self.epoch_frequency = epoch_frequency
        self.iterations = 1
        self.epochs = 1
        self.losses_by_iteration = []
        self.losses_by_epoch = []

    def on_batch_end(self, batch, logs={}):
        if self.iteration_frequency is not None:
            loss = logs["loss"]
            self.losses_by_iteration.append(loss)
            if self.iterations % self.iteration_frequency == 0:
                loss = '%.2f' % loss
                name = f"cp_it_{self.iterations}_loss_{loss}.h5"
                self.model.save_weights(os.path.join(self.output_dir, name))
                plt.plot(list(range(1, self.iterations+1)), self.losses_by_iteration)
                plt.title('training loss')
                plt.ylabel('loss')
                plt.xlabel('iteration')
                plt.savefig(os.path.join(self.output_dir, "log.png"))
        self.iterations += 1

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch_frequency is not None:
            loss = logs["loss"]
            self.losses_by_epoch.append(loss)
            if self.epochs % self.epoch_frequency == 0:
                loss = '%.2f' % loss
                name = f"cp_ep_{self.epochs}_loss_{loss}.h5"
                self.model.save_weights(os.path.join(self.output_dir, name))
                plt.plot(list(range(1, self.epochs+1)), self.losses_by_epoch)
                plt.title('training loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(self.output_dir, "log.png"))
        self.epochs += 1
