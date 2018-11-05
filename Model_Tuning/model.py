import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, timesteps, dimensions):
        self.model.add(LSTM(100, input_shape=(timesteps, dimensions), return_sequences=True))        
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(100, input_shape=(timesteps, dimensions), return_sequences=True))        
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(100, input_shape=(timesteps, dimensions), return_sequences=False))        
        self.model.add(Dropout(0.2))               
                
        self.model.add(Dense(1, activation="linear"))
        
        self.model.compile(loss="mse", optimizer="adam")
        print('[Model] Model Compiled')
      
    
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, save_name):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        save_fname = os.path.join(save_dir, '{}.h5'.format(save_name))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        train_history = self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return train_history

    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, xdata, window_size, prediction_len):
        #Predict sequence of n steps before shifting prediction run forward by n steps
        prediction_seqs = []
        for i in range(int(len(xdata)/prediction_len)):
            curr_frame = xdata[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]                               
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
