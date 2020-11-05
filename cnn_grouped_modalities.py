import os
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from tensorflow.keras.utils import to_categorical

from dataset import DataReader


checkpoint_path = "training_2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# lstm
def model_lstm():
    inputs = []
    hidden_states = []
    for position in ['Hips']:
        for _, channel in DataReader.channels.items():
            ts = keras.Input(shape=(500,), name=position + '_' + channel)
            x = layers.Reshape((500, 1))(ts)

            x = layers.LSTM(32,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-1, l2=1e-1),
                            recurrent_regularizer=regularizers.l2(1e-1),
                            bias_regularizer=regularizers.l2(1e-1),
                            activity_regularizer=regularizers.l2(1e-1),
                            # dropout,
                            # recurrent_dropout
                            )(x)

            inputs.append(ts)
            hidden_states.append(x)

    x = layers.concatenate(hidden_states)
    x = layers.Flatten()(x)

    x = layers.Dense(2048, activation='relu')(x)
    class_output = layers.Dense(8, activation='softmax', name='class_output')(x)

    model = keras.Model(inputs=inputs, outputs=class_output)

    keras.utils.plot_model(model, 'lstm_split_channels_model.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    print('[SHL Challenge] model compiled successfully')
    return model



# split channels
def model_split_channels():
    xs = []
    inputs = []
    feature_maps = []
    # for position in DataReader.smartphone_positions:
    for position in ['Hips']:
        for _, channel in DataReader.channels.items():
            ts = keras.Input(shape=(500,), name=position + '_' + channel)  # 3D tensor with shape: (batch_size, steps, input_dim)
            x = layers.Reshape((500, 1))(ts)
            # xs.append(x)

            x = layers.Conv1D(filters=32, kernel_size=13, strides=2, padding='valid', activation='relu', input_shape=(None, 500, 1))(x)
            x = layers.MaxPooling1D()(x)
            x = layers.BatchNormalization()(x)

            x = layers.Conv1D(filters=16, kernel_size=15, strides=2, padding='valid', activation='relu')(x)
            x = layers.MaxPooling1D()(x)
            x = layers.BatchNormalization()(x)

            x = layers.Conv1D(filters=8, kernel_size=9, strides=2, padding='valid', activation='relu')(x)
            x = layers.GlobalMaxPooling1D()(x)
            x = layers.BatchNormalization()(x)

            inputs.append(ts)
            feature_maps.append(x)

    x = layers.concatenate(feature_maps)  # , axis=-1)

    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    class_output = layers.Dense(8, activation='softmax', name='class_output')(x)

    model = keras.Model(inputs=inputs, outputs=class_output)

    keras.utils.plot_model(model, 'cnn_split_modalities_model.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    print('[SHL Challenge] model compiled successfully')
    return model



# grouped modalities (in this model, everything is grouped as opposed
# to split modalities models, which process groups of channels
# (corresponding to a same modality) individually
def model_grouped_modalities():
    xs = []
    inputs = []
    feature_maps = []
    # for position in DataReader.smartphone_positions:
    # for position in ['Hips']:
    for _, channel in DataReader.channels.items():
        ts = keras.Input(shape=(500,), name=channel)  # 3D tensor with shape: (batch_size, steps, input_dim)
        inputs.append(ts)
        x = layers.Reshape((500, 1))(ts)
        xs.append(x)

    x = layers.concatenate(xs, axis=-1)

    x = layers.Conv1D(filters=32, kernel_size=13, strides=2, padding='valid', activation='relu', input_shape=(None, 500, 20))(x)
    x = layers.MaxPooling1D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=16, kernel_size=15, strides=2, padding='valid', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=8, kernel_size=9, strides=2, padding='valid', activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    class_output = layers.Dense(8, activation='softmax', name='class_output')(x)
    print(class_output)

    model = keras.Model(inputs=inputs, outputs=class_output)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    keras.utils.plot_model(model, 'cnn_grouped_modalities_model.png', show_shapes=True)

    return model


def train():
    # training

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # load training data
    train = DataReader(what='train')
    train_dict = {}
    # for position in DataReader.smartphone_positions:
    for position in ['Hips', 'Torso']:
        for _, channel in DataReader.channels.items():
            if channel not in train_dict:
                train_dict[channel] = train.X[position][channel]
            else:
                train_dict[channel] = np.concatenate((train_dict[channel], train.X[position][channel]))

    print('shape of train_dict[''Gyr_x''] = ', train_dict['Gyr_x'].shape)

    train_labels = train.y[:, 0] - 1
    train_labels = np.repeat(train_labels,
                             # repeats=len(DataReader.smartphone_positions),
                             repeats=2,
                             axis=0)
    print('distribution of train_labels = ', np.bincount(train_labels))
    train_labels = to_categorical(train_labels, num_classes=8)  # classes should be from 0 to num_classes-1
    print('shape of train_labels = ', train_labels.shape)

    model = model_grouped_modalities()

    hist = model.fit(train_dict, train_labels, shuffle=True,
                     validation_split=0.3, batch_size=256, epochs=130,
                     callbacks=[cp_callback])
    print('\nhistory dict:', hist.history)


def validate(pos='Hips', checkpoint_path=checkpoint_path):
    """
    Validate trained model on data generated by position `pos` (default: 'Hips')
    1. data generated by position `pos` are loaded;
    2. weights of the trained model are loaded from `checkpoint_path` (default: 'training_2/cp.ckpt');
    3. model is evaluated on the loaded data.
    """

    # validation
    # load validation data
    validation = DataReader(what='validation')
    validation_dict = {}
    for _, channel in DataReader.channels.items():
        validation_dict[channel] = validation.X[pos][channel]
    print('shape of validation_dict[''Gyr_x''] = ', validation_dict['Gyr_x'].shape)

    validation_labels = validation.y[:, 0] - 1
    print('distribution of validation_labels = ', np.bincount(validation_labels))
    validation_labels = to_categorical(validation_labels, num_classes=8)  # classes should be from 0 to num_classes-1
    print('shape of validation_labels = ', validation_labels.shape)

    model = model_grouped_modalities()

    # loss, acc = model.evaluate(validation_dict, validation_labels,
    #                            batch_size=32)
    # print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

    model.load_weights(checkpoint_path)

    # channels_dict = validation.X['Torso']
    # labels = to_categorical(validation.y[:, 0], num_classes=8)
    print('\n# Evaluate on validation data')
    loss, acc = model.evaluate(validation_dict, validation_labels,
                               batch_size=32)
    # print('validation loss, validation acc:', results)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    train()
    validate(pos='Hips', checkpoint_path='training_2/cp.ckpt')
