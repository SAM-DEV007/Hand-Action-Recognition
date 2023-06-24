from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

import os
import csv


if __name__ == '__main__':
    dataset = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Dataset\\', 'Dataset.csv')
    model_save = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))) + '\\Model_Data\\', 'Model.h5')

    x_dataset = []
    with open(dataset, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            r = list(map(float, row[2:]))
            x_dataset.append(r)

    x_dataset = tf.keras.utils.pad_sequences(x_dataset, padding="post", dtype='float32', maxlen=1000)
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.8, random_state = 42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value=0, input_shape=(1000, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.summary()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save, verbose=1, save_weights_only=False)
    es_callback = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

    model.save(model_save, include_optimizer=False)
