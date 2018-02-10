"""
This script

Based on: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
"""

import numpy as np
import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

import os
from matplotlib import pyplot as plt


def normalize_by_column(float_data):
    """ Normalize every column
    Each timeseries in the data is on a different scale (e.g. temperature, pressure measured in mbar).
    This normalizes each timeseries independently so that they all take small values on a similar scale.

    :param
    float_data: numpy matrix, all data points

    :return
    float_data: each column normalized
    """

    mean = float_data[:].mean(axis=0)
    float_data -= mean
    std = float_data[:].std(axis=0)
    float_data /= std

    return float_data


def read_file(show_graph = False):
    # data_dir = '/home/ubuntu/data/'
    fname = os.path.join('data/Daily_Temperatures_Jena.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[368:]

    try:
        float_data = np.zeros((len(lines), len(header) - 1))
        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(',')[1:]]
            float_data[i, :] = values

    except :
        pass

    # replace -999 values with 0
    float_data[float_data==-999] = 0

    if show_graph:
        temp = float_data[:, 3]  # temperature (in degrees Celsius)
        plt.plot(range(len(temp)), temp)
        plt.show()

    # Normalize
    # float_data = normalize_by_column(float_data[:])

    # Ignore year, month, day
    float_data = float_data[ : , 3: ]
    print('floatdata shape',float_data.shape)

    return float_data


def evaluate_naive_method(val_gen = None, val_steps=1):
    """ Evaluate our baseline using naive method

    :param val_gen:
    :param val_steps:
    :return:
    """
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 0]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('Naive prediction Mean Absolute Error:', np.mean(batch_maes))


def generator( data, lookback, delay=15, min_index=0, max_index=None,
              shuffle=False, batch_size=128, step=7):
    """Generator based on this data

    :param
    lookback = 60, i.e. our observations will go back (days).
    steps = 7, i.e. our observations will be sampled at one data point per week.
    delay = 15, i.e. our targets will be N days in the future.

    :return:
    """

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets


def models(name, float_data):
    if name == 'simple_gru':
        model = Sequential()
        model.add(layers.GRU(32,
                             dropout=0.1,
                             recurrent_dropout=0.5,
                             return_sequences=True,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.GRU(64, activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(1))

    if name == '1layer_LSTM':
        model = Sequential()
        model.add(layers.LSTM(32,
                             dropout=0.1,
                             recurrent_dropout=0.5,
                             return_sequences=True,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.LSTM(64, activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(1))

    return model


def main():
    # Load file
    float_data = read_file()

    # each sample has this week + 56 past weeks(~one year)
    lookback = 365  # one year
    step = 1    # do not skip data points (step over)
    delay = 15     # predict 15th day in the future
    batch_size = 100

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=15000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=15001,
                        max_index=18000,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=18001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (18000 - 15001 - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - 18001 - lookback) // batch_size

    evaluate_naive_method(val_gen=val_gen, val_steps=val_steps)

    for model_name in ['simple_gru', '1layer_LSTM']:
        print('Starting: ', model_name)
        model = None
        model = models(model_name, float_data)
        model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])

        history = model.fit_generator(train_gen,
                                      steps_per_epoch=15000//batch_size,
                                      epochs=2,
                                      validation_data=val_gen,
                                      validation_steps=val_steps, verbose=2)

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()






if __name__ == '__main__':
    main()