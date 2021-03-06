import contextlib
import os
import wave

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pathlib import Path
import librosa

avg_1_sec = 44576
labels = 3  # number of classes
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


def extract_features_avg(wav_file, path_to_wav):
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

#The function creates a csv contains avg mfcc data for all .wav at data_path
def make_avg_mfcc(data_path, path_to_csv):
    path_list = Path(data_path).glob('**/*.wav')
    with open(path_to_csv, 'w') as f:
        for path in path_list:
            path_to_wav = str(path)  # convert object path to string
            arr_feat = extract_features_avg(path_to_wav, path_to_wav)  # Features array
            for i in range(len(arr_feat) - 1):
                f.write("%s," % str(arr_feat[i]))
            f.write("%s\n" % str(arr_feat[len(arr_feat) - 1]))


# read in wav file, get out signal (np array) and sampling rate (int)
def read_in_audio(filename):
    (rate, sig) = wav.read(filename)
    return sig, rate


# Cut two seconds from the start, 1 second from the end
def cut_beg_end(filename):
    sig, rate = read_in_audio(filename)
    end = len(sig) - avg_1_sec
    return sig[avg_1_sec * 2:end]


# read in signal, change sample rate to outrate (samples/sec), use write_wav=True to save wav file to disk
def downsample(filename, outrate=8000, write_wav=False):
    (rate, sig) = wav.read(filename)
    sig = cut_beg_end(filename)  # Slice seconds 1 to 2 from the beginning, last second from the end
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


def write_mat_to_csv(ind, result, path_to_folder):
    with open(path_to_folder + str(ind) + ".csv", 'w') as f:  # Write result item into file
        for i in range(len(result)):
            for j in range(len(result[i]) - 1):
                f.write("%s," % str(result[i][j]))
            f.write("%s\n" % str(result[i][len(result[i]) - 1]))


def write_prediction_to_csv(ind, data, path_to_folder):
    with open(path_to_folder + str(ind) + "_pred.csv", 'w') as f:  # Write prediction into file
        for i in range(np.size(data, 0)):
            f.write("%s," % str(data[i]))


def write_data_to_csv(i, result, folder, target_folder, filename):
    write_mat_to_csv(i, result, target_folder)
    resolve_prediction(folder + "\\" + filename)
    data = np.zeros(shape=(1, 0))
    for j in range(labels):
        if j == classes_ind:
            data = np.append(data, 1)  # This is the class we predict
        else:
            data = np.append(data, 0)  # Rest of the classes will receive prediction 0
    write_prediction_to_csv(i, data, target_folder)


# for folder containing wav files, output numpy array of normed mfcc
def make_class_array(folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    lst = []
    i = 0
    for filename in os.listdir(folder):
        result = make_normed_mfcc(folder + "\\" + filename)
        lst.append(result)
        i += 1
        print("Done ", i)
    class_array = np.array(lst)
    class_array = np.reshape(class_array, (class_array.shape[0], class_array.shape[2], class_array.shape[1]))
    i = 0
    # Write output normed mfcc to csv
    for filename in os.listdir(folder):
        write_data_to_csv(i, class_array[i], folder, target_folder, filename)
        i += 1
    return class_array


def get_record_duration(filename):
    with contextlib.closing(wave.open(filename, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / (rate)
        return int(duration)


def get_avg_1_second(folder):
    sum = 0
    counter = 0
    for filename in os.listdir(folder):
        len_sig = len(read_in_audio(folder + "\\" + filename)[0])
        seconds = get_record_duration(folder + "\\" + filename)
        sum += len_sig / seconds
        counter += 1

    print("Avg 1 second is: ", sum / counter)


if __name__ == "__main__":
    # path to the training set
    train_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\train_set'
    # path to the test set
    test_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\test_set'

    # chooser = True  # False for train, True for test
    # data_path = train_set_path
    # path_to_csv = "resources/train_set_features.csv"
    # if chooser:
    #     data_path = test_set_path
    #     path_to_csv = "resources/test_set_features.csv"
    # make_avg_mfcc(data_path, path_to_csv)

    target_folder = "resources/normed_features/86_train_13_test_3_class_sliced_test/"
    make_class_array(test_set_path, target_folder)
