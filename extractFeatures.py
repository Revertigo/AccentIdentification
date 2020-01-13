import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pathlib import Path

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


if __name__ == "__main__":

    chooser = False # False for train, True for test
    # path to the training set
    train_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\train_set'
    # path to the test set
    test_set_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\test_set'

    data_path = train_set_path
    path_to_csv = "resources/train_set_features.csv"
    if chooser:
        data_path = test_set_path
        path_to_csv = "resources/test_set_features.csv"

    path_list = Path(data_path).glob('**/*.wav')
    with open(path_to_csv, 'w') as f:
        for path in path_list:
            path_to_wav = str(path)  # convert object path to string
            arr_feat = extract_features(path_to_wav, path_to_wav)  # Features array
            for i in range(len(arr_feat) - 1):
                f.write("%s," % str(arr_feat[i]))
            f.write("%s\n" % str(arr_feat[len(arr_feat) - 1]))
