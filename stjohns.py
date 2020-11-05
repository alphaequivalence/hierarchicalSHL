import logging

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import nni

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import Normalizer

from dataset import DataReader
from balancedsample import BalancedSample
from refinedsample import RefinedSample

_logger = logging.getLogger('[SHL Challenge]')
_logger.setLevel(logging.INFO)


tf.random.set_seed(1234)
np.random.seed(1234)

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def stjohns(params, num_classes=None):
    # split channels
    # + modality-dependent hyperparameters (a kind of bias selection)
    # + considering all channels from all positions
    if num_classes is None:  # ensure back compatibility
        num_classes = 8
    xs = []
    inputs = []
    all_views = []
    for position in DataReader.smartphone_positions:
        one_view = []
        for _, channel in DataReader.channels.items():
            print('circuit of channel {}'.format(channel))
            modality = DataReader.channel_to_modality(channel)

            if channel.startswith('Acc') and channel.endswith('spectralfeatures'):
                input_length=60
            elif channel == 'Mag_spectralfeatures':
                input_length=73
            else:
                input_length=500

            # 3D tensor with shape: (batch_size, steps, input_dim)
            ts = keras.Input(shape=(input_length,), name=position+'_'+channel)
            x = layers.Reshape((input_length, 1))(ts)

            # xs.append(x)  # this is for grouped modalities

            # x = layers.Dense(
            #     # 8,  # as the number of classes. In order to be used as output layer during validation and test!
            #     # activation='softmax',
            #     64,
            #     activation='relu',
            #     kernel_regularizer=regularizers.l2(0.001),
            #     bias_regularizer=regularizers.l2(0.001),
            #     activity_regularizer=regularizers.l2(0.001),
            #     # name='view_'+position
            # )(x)
            # # x = layers.Dropout(params['All']['dropout']['3'])(x)
            # x = layers.Dropout(0.5)(x)

            # x = layers.LSTM(
            #     32,
            #     kernel_regularizer=regularizers.l1_l2(l1=1e-1, l2=1e-1),
            #     recurrent_regularizer=regularizers.l2(1e-1),
            #     bias_regularizer=regularizers.l2(1e-1),
            #     activity_regularizer=regularizers.l2(1e-1),
            #     # dropout,
            #     # recurrent_dropout
            # )(x)

            x = layers.Conv1D(
                filters=params[modality]['numfilters']['0'],
                kernel_size=params[modality]['kernelsize']['0'],
                strides=2,
                padding='valid',
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l2(0.001),
                # input_shape=(None, 500, 1),
                input_shape=(None, input_length, 1),
                name=position+'/'+channel+'/Conv1d/layer_0')(x)
            x = layers.MaxPooling1D()(x)
            x = layers.BatchNormalization(name=position+'/'+channel+'/BN/layer_0')(x)

            x = layers.Conv1D(
                filters=params[modality]['numfilters']['1'],
                kernel_size=params[modality]['kernelsize']['1'],
                strides=2,
                padding='valid',
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l2(0.001),
                name=position+'/'+channel+'/Conv1d/layer_1')(x)
            x = layers.MaxPooling1D()(x)
            x = layers.BatchNormalization(name=position+'/'+channel+'/BN/layer_1')(x)

            x = layers.Conv1D(
                 filters=params[modality]['numfilters']['2'],
                 kernel_size=params[modality]['kernelsize']['2'],
                 strides=2,
                 padding='valid',
                 activation='relu',
                 kernel_regularizer=regularizers.l2(0.001),
                 bias_regularizer=regularizers.l2(0.001),
                 activity_regularizer=regularizers.l2(0.001),
                 name=position+'/'+channel+'/Conv1d/layer_2')(x)
            #---------------------------------
            x = layers.GlobalMaxPooling1D()(x)
            #---------------------------------
            x = layers.BatchNormalization(name=position+'/'+channel+'/BN/layer_3')(x)

            inputs.append(ts)
            one_view.append(x)

        x = layers.concatenate(one_view)
        x = layers.Dense(
            # 8,  # as the number of classes. In order to be used as output layer during validation and test!
            # activation='softmax',
            params['All']['viewReprDim']['3'],
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            activity_regularizer=regularizers.l2(0.001),
            name='view_'+position)(x)
        # x = layers.Dropout(params['All']['dropout']['3'])(x)
        x = layers.Dropout(0.5)(x)
        all_views.append(x)

    joint_representation = layers.concatenate(all_views, name='joint_representation')

    joint_representation = layers.Dense(
        units=params['All']['hiddenunits']['3'],
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        bias_regularizer=regularizers.l2(0.001),
        activity_regularizer=regularizers.l2(0.001))(joint_representation)
    joint_representation = layers.Dropout(params['All']['dropout']['3'])(joint_representation)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(joint_representation)

    model = keras.Model(inputs=inputs, outputs=class_output)

    keras.utils.plot_model(model, 'stjohns.png', show_shapes=True)

    return model


def dict_of_params(params):
    """
    Transforms a flattened dictionary, given in `params`, into
    an easy-to-access nested dictionary, e.g.:
    input:
        params = {
            'Acc_numfilters_0': 32
            [...]
        }

    result:
        params = {
            'Acc': {
                'numfilters': {
                        '0': 32,
                        '1': 9
                }
            [...]
        }
    """
    ret = {}
    for key, val in params.items():
        a, b, c = key.split('_')
        if a not in ret:
            ret[a] = {}
        if b not in ret[a]:
            ret[a][b] = {}
        ret[a][b][c] = val
    return ret


#callbacks
#checkpoint_path = "{}/{}/cp.ckpt".format(nni.get_experiment_id(), nni.get_sequence_id())
checkpoint_path = "stjohns_47/cp.{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq='epoch')


class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.

    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.

    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""

        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return tf.keras.backend.eval(optimizer.lr)
#     return lr


def load_data(what='train', new_classes=None, refinedSampleId=None):
    data = DataReader(what=what)
    # data = BalancedSample(id='stjohns')
    if refinedSampleId is not None and new_classes is not None:
        data = RefinedSample(new_classes, datareader=data, id=refinedSampleId)

    data.normalize_NOT_in_place()

    _dict = {}
    for position in DataReader.smartphone_positions:
        for _, channel in DataReader.channels.items():
            _dict[position + '_' + channel] = data.X[position][channel]

    if refinedSampleId is not None:
        _labels = data.y  # labels are already starting from 0 and are of shape (num_examples,)
    else:
        _labels = data.y[:, 0] - 1

    return _dict, _labels

# def load_validation_data(what='validation'):
#     data = DataReader(what=what)
#     # normalizer = Normalizer(copy=False)  # <--- BEWARE: this will perform in-place normalization!
#
#     _dict = {}
#     for position in DataReader.smartphone_positions:
#         for _, channel in DataReader.channels.items():
#             _dict[position + '_' + channel] = data.X[position][channel]
#
#     _labels = data.y[:, 0] - 1
#
#     return _dict, _labels


def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    #map original labels to the new learning problem
    new_classes = {
            0: ['bike'],
            1: ['car'],
            2: ['bus']
    }
    num_classes = len(list(new_classes.keys()))
    refinedSampleId = 'BikeCarBus'


    params = dict_of_params(params)
    model = stjohns(params, num_classes=num_classes)
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)  # , amsgrad=True)
    # lr_metric = get_lr_metric(optimizer)
    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # , amsgrad=True),
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc'
                 # , lr_metric
                 ])
    _logger.info('Model built')

    # (x_train, y_train), (x_test, y_test) = load_dataset()
    train_dict, train_labels = load_data(what='train', new_classes=new_classes, refinedSampleId=refinedSampleId)
    print('bincount(train_labels) = {}'.format(np.bincount(train_labels)))
    _logger.info('Train data loaded')

    validation_dict, validation_labels = load_data(what='validation', new_classes=new_classes, refinedSampleId=refinedSampleId)
    print('bincount(validation_labels) = {}'.format(np.bincount(validation_labels)))
    _logger.info('Validation data loaded')

    history = model.fit(
        train_dict,
        to_categorical(train_labels, num_classes=num_classes),
        batch_size=512,
        epochs=50,
        callbacks=[
            ReportIntermediates(),
            cp_callback
        ],
        # validation_split=0.3,
        validation_data=(validation_dict, to_categorical(validation_labels, num_classes=num_classes)),
        shuffle=True,
        class_weight={0: 2.,
                      1: 1.,  # give class run 10 times the weight of class 0
                      2: 4.
                      }
    )
    _logger.info('Training completed')


    score = {}
    score['default'] = tf.math.reduce_mean(history.history['val_acc'][-5:]).numpy()
    print(score)
    print(history.history['acc'])
    print(history.history['val_acc'])
    nni.report_final_result(score)
    _logger.info('Final accuracy reported: %s', score)


params = {  # arch#184 (SaFpz) gave 66.54% on validation (tehran_1)
    'Acc_numfilters_0': 64,
    'Acc_kernelsize_0': 11,
    'Acc_numfilters_1': 32,
    'Acc_kernelsize_1': 5,
    'Acc_numfilters_2': 32,
    'Acc_kernelsize_2': 2,

    'Gyr_numfilters_0': 64,
    'Gyr_kernelsize_0': 17,
    'Gyr_numfilters_1': 8,
    'Gyr_kernelsize_1': 17,
    'Gyr_numfilters_2': 16,
    'Gyr_kernelsize_2': 5,

    'Mag_numfilters_0': 16,
    'Mag_kernelsize_0': 7,
    'Mag_numfilters_1': 16,
    'Mag_kernelsize_1': 17,
    'Mag_numfilters_2': 16,
    'Mag_kernelsize_2': 2,

    'Ori_numfilters_0': 32,
    'Ori_kernelsize_0': 7,
    'Ori_numfilters_1': 32,
    'Ori_kernelsize_1': 5,
    'Ori_numfilters_2': 16,
    'Ori_kernelsize_2': 5,

    'Gra_numfilters_0': 32,
    'Gra_kernelsize_0': 11,
    'Gra_numfilters_1': 8,
    'Gra_kernelsize_1': 7,
    'Gra_numfilters_2': 16,
    'Gra_kernelsize_2': 2,

    'LAc_numfilters_0': 32,
    'LAc_kernelsize_0': 17,
    'LAc_numfilters_1': 32,
    'LAc_kernelsize_1': 17,
    'LAc_numfilters_2': 16,
    'LAc_kernelsize_2': 17,

    'Pre_numfilters_0': 8,
    'Pre_kernelsize_0': 11,
    'Pre_numfilters_1': 16,
    'Pre_kernelsize_1': 11,
    'Pre_numfilters_2': 16,
    'Pre_kernelsize_2': 2,

    'All_viewReprDim_3': 10,
    'All_hiddenunits_3': 128,
    'All_dropout_3': 0.8154384601076919,
    # 'All_regularization_3': 0.001,  -> fixed
    # 'All_batchsize_3': 256,  -> fixed
    # 'All_learningrate_3': 0.001,  -> fixed
}



if __name__ == '__main__':
    # fetch hyper-parameters from HPO tuner
    # comment out following two lines to run the code without NNI framework
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)

    _logger.info('Hyper-parameters: %s', params)
    main(params)
