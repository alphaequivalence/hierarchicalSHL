import os
import numpy as np
import pickle

from config import Configuration as config

from dataset import DataReader


class RefinedSample(object):
    """
    Creates a sample that group some classes together for the purpose
    of model refinement.

    Example:
    ```python
    from dataset import DataReader

    train = DataReader(what='train')
    sample = Sample(datareader=train, id='cairo')
    print('Shape of sample.X[''Acc_x'']')
    print(sample.X['Acc_x'].shape)
    print('Shape of sample.y')
    print(sample.y.shape)    ```
    """
    def __init__(self, new_classes, datareader=None, id='ulanbator'):
        # self.datareader = datareader  # do not keep it! memory issues!
        self.new_classes = new_classes

        self.what = datareader.what
        self.id = id

        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        # e.g. generated/0.1/SampleSelection/
        self.sample_path = os.path.join(
            config.experimentsfolder,
            'SampleSelection',
            self.id,
            self.what)
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)

        # create an intermediate data strucutre to store data
        # self._create_temp_data_structure()

        # self._data = self._load_data(what)
        # self._labels = self._load_labels(what)
        self._examples, self._labels = self._create_sample(datareader)

    @property
    def X(self):
        return self._examples

    @property
    def y(self):
        return self._labels

    def _save_obj(self, obj, dest):
        with open(dest, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self, src):
        with open(src, 'rb') as f:
            return pickle.load(f)

    def _create_sample(self, datareader):

        #---------------------------------------------------------
        # This is the part that changes between sample strategies
        # TODO: make the remaining of the class as a super class
        # and make this part proper to each sample strategie.
        if datareader is not None:
            X = datareader.X
            y = datareader.y

            original_labels = y[:, 0]

            num_examples = 0
            sampling_indices = []
            new_labels = np.zeros_like(original_labels)
            for new_label, klasses in self.new_classes.items():
                for klass in klasses:
                    label = DataReader.inv_coarselabel_map[klass]
                    index = np.argwhere(original_labels == label)
                    print('index.shape = {}'.format(index.shape))
                    num_examples += index.shape[0]
                    new_labels[index] = new_label
                    sampling_indices.append(index[:, 0])
        #---------------------------------------------------------
        print('num_examples = {}'.format(num_examples))

        examples = {}
        y = new_labels

        for position in DataReader.smartphone_positions:
            if position not in examples:
                examples[position] = {}

            for _, channel in DataReader.channels.items():

                key = \
                    position + '_' +\
                    channel

                dest = os.path.join(
                    self.sample_path,
                    key + '.mmap')

                # FIXME
                if channel.startswith('Acc') and channel.endswith('spectralfeatures'):
                    samples = 60
                elif channel == 'Mag_spectralfeatures':
                    samples = 73
                else:
                    samples = 500

                shape = (num_examples, samples)

                if os.path.exists(dest):
                    # just load mmap file contents
                    print('%s exists, loading ...' % dest)
                    examples[position][channel] = np.memmap(
                        dest,
                        mode='r+',
                        dtype=np.double,
                        shape=shape)

                else:
                    # build mmap file from scratch
                    print('Building from scratch %s ...' % dest)
                    print(shape)
                    examples[position][channel] = np.memmap(
                        dest,
                        mode='w+',
                        dtype=np.double,
                        shape=shape)

                    offset = 0
                    for one_sampling_index in sampling_indices:
                        e = np.array(X[position][channel][one_sampling_index])
                        examples[position][channel][offset:offset+one_sampling_index.shape[0]] = e
                        offset += one_sampling_index.shape[0]

                    print('examples[{}][{}].shape = {}'
                          .format(position, channel, examples[position][channel].shape))

                    examples[position][channel].flush()


        shape = (num_examples,)  #FIXME has to be (num_examples, 500)
        dest = os.path.join(
            self.sample_path,
            'Label.mmap')

        if os.path.exists(dest):
            # just load mmap file contents
            print('%s exists, loading ...' % dest)
            labels = np.memmap(
                dest,
                mode='r+',
                dtype=np.integer,
                shape=shape)

        else:
            # build mmap file from scratch
            print('Building from scratch %s ...' % dest)
            print(shape)
            labels = np.memmap(
                dest,
                mode='w+',
                dtype=np.integer,
                shape=shape)

            offset = 0
            for one_sampling_index in sampling_indices:
                labels[offset:offset+one_sampling_index.shape[0]] = np.array(y[one_sampling_index])
                offset += one_sampling_index.shape[0]

            print('labels.shape = {}'.format(labels.shape))
            labels.flush()

        return examples, labels

    def normalize_NOT_in_place(self):
        from sklearn.preprocessing import Normalizer
        normalizer = Normalizer(copy=True)  # <--- BEWARE: if you put False, this will normalize in-place and produce a side-effect
        if self.what == 'test':
            for _, channel in DataReader.channels.items():
                self.X[channel] = normalizer.transform(self.X[channel])
                # self.X[channel].flush()
        else:
            for position in DataReader.smartphone_positions:
                for _, channel in DataReader.channels.items():
                    self.X[position][channel] = normalizer.transform(self.X[position][channel])
                    # self.X[position][channel].flush()

    def CHECK_wholeFrameIsZeroed(self):
        # checking for nan's
        zeros = {}
        for position in DataReader.smartphone_positions:
            for _, channel in DataReader.channels.items():
                for i, a in enumerate(self.X[position][channel]):
                    if (a==0).all():
                        if position not in zeros:
                            zeros[position] = {}
                        if channel not in zeros[position]:
                            zeros[position][channel] = []
                        zeros[position][channel].append(i)

        print('there are frames completely zeroed')
        return zeros

    def CHECK_transition_frames(self):
        """
        This function checks for frames that contain a transition between two activities.

        Returns
         a list containing the index of the frames that contain a transition between two activities
        """
        tr_frames = []
        for i, frame in enumerate(self.y):
            if not np.all(frame == frame[0]):
                tr_frames.append(frame)

        print('there are ', len(tr_frames), ' frames containing a transition')
        return tr_frames


if __name__ == '__main__':
    from dataset import DataReader

    train = DataReader(what='train')
    # sample = Sample(datareader=train, id='cairo')
    # print('Shape of sample.X[''Acc_x'']')
    # print(sample.X['Acc_x'].shape)
    # print('Shape of sample.y')
    # print(sample.y.shape)

    bal_sample = BalancedSample(datareader=train, id='ljubljana')
    print('shape of bal_sample.X[{}][{}] = {}'
          .format('Hips', 'Acc_x', bal_sample.X['Hips']['Acc_x'].shape))
