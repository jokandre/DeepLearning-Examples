"""
This script

Based on: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
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
        preds = samples[:, -1, 1]
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
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def main():
    # Load file
    float_data = read_file()

    # each sample has this week + 56 past weeks(~one year)
    lookback = 56 # one year
    step = 7    # one week has 7 days of data(one per day)
    delay = 2     # week
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



    max_features = 10000  # number of words to consider as features
    maxlen = 500  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    # history = model.fit(input_train, y_train,
    #                     epochs=1,
    #                     batch_size=128,
    #                     validation_split=0.2)


if __name__ == '__main__':
    main()