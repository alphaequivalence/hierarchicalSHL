import os
import numpy as np
import scipy
import acoustics

from config import Configuration as config


class SpectralFeatures():
    def __init__(self, datareader):
        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        # e.g. generated/0.1/train/ or generation/0.1/validation
        path = os.path.join(
            config.experimentsfolder,
            datareader.what)
        if not os.path.exists(path):
            os.makedirs(path)

        self.extractFeatures(datareader)

    def subBandEnergy(self, spectrum):
        sb_energy = []
        for band in np.array_split(spectrum, 10):
            sb_energy.append(band.sum())
        return np.array(sb_energy)

    def subBandEnergyRatio(self, sbEnergy, spectrumEnergy):
        #spectrum_energy = spectrum.sum()
        sb_energy_ratio = []
        for sb in sbEnergy:
            if spectrumEnergy == 0:
                sb_energy_ratio.append(0)
            else:
                sb_energy_ratio.append(sb/spectrumEnergy)
        return np.array(sb_energy_ratio)

    def subBandSpectralEntropy(self, spectrum):
        sb_spectral_entropy = []
        for band in np.array_split(spectrum, 10):
            #print(band)
            sb_spectral_entropy.append(scipy.stats.entropy(band))
        return np.array(sb_spectral_entropy)

    def cepstralCoefficients(self, x):
        ceps = acoustics.cepstrum.real_cepstrum(x)
        if np.isnan(ceps).any():
            return np.zeros_like(ceps)
        return ceps

    def firstOrderDifference(self, ceps):
        return np.diff(ceps)

    def extractFeatures(self, datareader):
        """
        Extract features from Acc and Mag data
        and stores them in mmaped files as new
        channels alondside the rest of channels.
        """

        self.extractAccSpectralFeatures(datareader)
        self.extractMagSpectralFeatures(datareader)

    def extractAccSpectralFeatures(self, datareader):

        for position in datareader.smartphone_positions:
            # Acc data (features are extracted from the spectrum of each channel:
            for channel in ['Acc_x', 'Acc_y', 'Acc_z']:
                new_channel = channel+'_spectralfeatures'
                num_features = 60

                shape = (datareader.y.shape[0], num_features)  # 64 in the original paper but firstOrderDifference returns #ceps-1 and energy of the 1Hz is not included here

                key = \
                    datareader.what + '_' +\
                    position + '_' +\
                    new_channel

                dest = os.path.join(
                    config.experimentsfolder,
                    datareader.what,
                    key + '.mmap')

                if os.path.exists(dest):
                    print('%s exists, no further steps required')
                else:
                    # build mmap file from scratch
                    print('Building from scratch %s ...' % dest)
                    print(shape)
                    mmap = np.memmap(
                        dest,
                        mode='w+',
                        dtype=np.double,
                        shape=shape)

                    # chunksize = 5000
                    # offset = 0
                    # for chunk in pd.read_csv(src, delimiter=' ', chunksize=chunksize, header=None):
                    #     mmap[offset:offset+chunk.shape[0]] = chunk.values
                    #     offset += chunk.shape[0]

                    for i, x in enumerate(datareader.X[position][channel]):
                        spectrum = acoustics.signal.power_spectrum(x, fs=100, N=500)[1]  # N=500 number of FFT bins
                        features = np.zeros((num_features,))

                        spectrumEnergy = spectrum.sum()
                        features[0] = spectrumEnergy  #1

                        #energy of the 1Hz component (not computed) #2

                        sbEnergy = self.subBandEnergy(spectrum)
                        features[1:11] = sbEnergy  #3~12

                        sbEnergyRatio = self.subBandEnergyRatio(sbEnergy, spectrumEnergy)
                        features[11:21] = sbEnergyRatio  #13~22

                        ceps = self.cepstralCoefficients(x)[1:21]
                        features[21:41] = ceps  #23~42

                        features[41:60] = self.firstOrderDifference(ceps)  #43~62

                        mmap[i] = features

                    mmap.flush()


    def extractMagSpectralFeatures(self, datareader):
        # Mag data (features are extracted from a unique spectrum which is the sum of the x/y/z spectrums

        for position in datareader.smartphone_positions:
            new_channel = 'Mag_spectralfeatures'
            num_features = 73
            shape = (datareader.y.shape[0], num_features)  # 74 in the original paper but firstOrderDifference returns #ceps-1

            key = \
                datareader.what + '_' +\
                position + '_' +\
                new_channel

            dest = os.path.join(
                config.experimentsfolder,
                datareader.what,
                key + '.mmap')

            if os.path.exists(dest):
                print('%s exists, no further steps required')
            else:
                # build mmap file from scratch
                print('Building from scratch %s ...' % dest)
                print(shape)
                mmap = np.memmap(
                    dest,
                    mode='w+',
                    dtype=np.double,
                    shape=shape)

                # chunksize = 5000
                # offset = 0
                # for chunk in pd.read_csv(src, delimiter=' ', chunksize=chunksize, header=None):
                #     mmap[offset:offset+chunk.shape[0]] = chunk.values
                #     offset += chunk.shape[0]

                X = datareader.X[position]['Mag_x']
                Y = datareader.X[position]['Mag_y']
                Z = datareader.X[position]['Mag_z']
                for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                    spectrum_x = acoustics.signal.power_spectrum(x, fs=100, N=500)[1]  # N=500 number of FFT bins
                    spectrum_y = acoustics.signal.power_spectrum(y, fs=100, N=500)[1]  # N=500 number of FFT bins
                    spectrum_z = acoustics.signal.power_spectrum(z, fs=100, N=500)[1]  # N=500 number of FFT bins

                    spectrum = spectrum_x + spectrum_y + spectrum_z

                    features = np.zeros((num_features,))

                    features[0] = np.max(spectrum)  #1
                    features[1] = np.argmax(spectrum)  #2
                    features[2] = scipy.stats.entropy(spectrum)  #3
                    spectrumEnergy = spectrum.sum()
                    features[3] = spectrumEnergy  #4

                    sbEnergy = self.subBandEnergy(spectrum)
                    features[4:14] = sbEnergy  #5~14

                    sbEnergyRatio = self.subBandEnergyRatio(sbEnergy, spectrumEnergy)
                    features[14:24] = sbEnergyRatio  #15~24

                    features[24:34] = self.subBandSpectralEntropy(spectrum)  #25~34

                    # ceps_x = self.cepstralCoefficients(x)[1:21]
                    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real[1:21]  # as it is for a spectrum constructed from 3 different channels
                    features[34:54] = ceps  #35~54

                    features[54:73] = self.firstOrderDifference(ceps)  #55~74

                    mmap[i] = features
                    # print('{}: {}'.format(i, mmap[i]))

                mmap.flush()


if __name__ == '__main__':
    from dataset import DataReader
    train = DataReader(what='train')
    SpectralFeatures(datareader=train)
