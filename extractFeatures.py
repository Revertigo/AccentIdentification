import os

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pathlib import Path
import librosa

labels = 44  # number of classes
classes_ind = -1
last_file_name = ''


def resolve_prediction(path_to_wav):
    global classes_ind
    global last_file_name
    # Just because split in python can't work with more then 1 delimiter !!!
    temp_path_str = path_to_wav.replace('.', '\\')
    # Extract file name with extension
    tokens = temp_path_str.split('\\')  # by slash
    file_name = tokens[-2]  # second from the last token is the file name
    file_name = ''.join(filter(str.isalpha, file_name))  # extract only letters

    if file_name != last_file_name:
        classes_ind += 1

    last_file_name = file_name


def extract_features(wav_file, path_to_wav):
    """
    The function extracts the features from the wav_file uses as recording file.
    @:param wav_file: Input file to extract features from
    @:param path_to_wav: full path to the recording

    @:returns: The function returns a numpy array of features including the prediction concatenated to the end
    """
    (rate, sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig, rate, winfunc=np.hamming, nfft=2048)

    # Write mfcc output features into one vector contains all features.
    data = np.mean(mfcc_feat, axis=0)  # average all columns

    resolve_prediction(path_to_wav)  # function has side effects
    for i in range(labels):
        if i == classes_ind:
            data = np.append(data, 1)  # This is the class we predict
        else:
            data = np.append(data, 0)  # Rest of the classes will receive prediction 0

    return data


# read in signal, change sample rate to outrate (samples/sec), use write_wav=True to save wav file to disk
def downsample(filename, outrate=8000, write_wav=False):
    (rate, sig) = wav.read(filename)
    sig = [float(i) for i in sig]
    sig = np.asarray(sig)
    down_sig = librosa.core.resample(sig, float(rate), outrate, scale=True)
    if not write_wav:
        return down_sig, outrate
    if write_wav:
        wav('{}_down_{}.wav'.format(filename, outrate), outrate, down_sig)


# change total number of samps for downsampled file to n_samps by trimming or zero-padding and standardize them
def make_standard_length(filename, n_samps=240000):
    down_sig, rate = downsample(filename)
    normed_sig = librosa.util.fix_length(down_sig, n_samps)
    normed_sig = (normed_sig - np.mean(normed_sig)) / np.std(normed_sig)
    return normed_sig


# for input wav file outputs (13, 2999) mfcc np array
def make_normed_mfcc(filename, outrate=8000):
    normed_sig = make_standard_length(filename)
    normed_mfcc_feat = mfcc(normed_sig, outrate)
    normed_mfcc_feat = normed_mfcc_feat.T
    return normed_mfcc_feat


# for folder containing wav files, output numpy array of normed mfcc
def make_class_array(folder):
    path_to_folder = "resources/train_set/"
    lst = []
    i = 0
    for filename in os.listdir(folder):
        result = make_normed_mfcc(folder + "\\" + filename)
        lst.append(result)
        write_mat_to_csv(i, result, path_to_folder)
        resolve_prediction(folder + "\\" + filename)
        data = np.zeros(shape=(1,0))
        for j in range(labels):
            if j == classes_ind:
                data = np.append(data, 1)  # This is the class we predict
            else:
                data = np.append(data, 0)  # Rest of the classes will receive prediction 0
        write_prediction_to_csv(i, data, path_to_folder)
        i += 1
        print("Done ", i)
    class_array = np.array(lst)
    class_array = np.reshape(class_array, (class_array.shape[0], class_array.shape[2], class_array.shape[1]))
    print(class_array.shape)
    return class_array

def write_mat_to_csv(ind, result, path_to_folder):
    with open(path_to_folder + str(ind) + ".csv", 'w') as f:  # Write result item into file
        for j in range(len(result) - 1):
            for k in range(len(result[j])):
                f.write("%s," % str(result[j][k]))
            f.write("%s\n" % str(result[j][len(result) - 1]))

def write_prediction_to_csv(ind, data, path_to_folder):
    with open(path_to_folder + str(ind) + "_pred.csv", 'w') as f:  # Write prediction into file
        for i in range(np.size(data, 0)):
            f.write("%s," % str(data[i]))


if __name__ == "__main__":
    chooser = False  # False for train, True for test
    # path to the training set
    train_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\train_set'
    # path to the test set
    test_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\test_set'
    lefties_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\lefties'

    # data_path = train_set_path
    # path_to_csv = "resources/train_set_features.csv"
    # if chooser:
    #     data_path = test_set_path
    #     path_to_csv = "resources/test_set_features.csv"
    #
    # path_list = Path(data_path).glob('**/*.wav')
    # with open(path_to_csv, 'w') as f:
    #     for path in path_list:
    #         path_to_wav = str(path)  # convert object path to string
    #         arr_feat = extract_features(path_to_wav, path_to_wav)  # Features array
    #         for i in range(len(arr_feat) - 1):
    #             f.write("%s," % str(arr_feat[i]))
    #         f.write("%s\n" % str(arr_feat[len(arr_feat) - 1]))

    make_class_array(train_set_path)
