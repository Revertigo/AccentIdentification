import os
from pathlib import Path

if __name__ == "__main__":
    split_train_factor = 0.85  # percentage
    #minimum_records = 60  # Minimum records of accents we allowed to train on - 5 classes
    minimum_records = 75  # Minimum records of accents we allowed to train on - 3 classes
    # Path to data set
    path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings'
    train_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\train_set'
    test_path = r'C:\Users\Dekel\Downloads\לימודים\deep_learning\datasets\speech-accent-archive\recordings\test_set'

    if not os.path.exists(path + '\\train_set'):
        os.makedirs(path + '\\train_set')

    if not os.path.exists(path + '\\test_set'):
        os.makedirs(path + '\\test_set')

    path_list = Path(path).glob('**/*.wav')

    all_data = []
    temp_accents_list = []
    last_file = ''
    for file in path_list:  # file is a full path to .wav file
        path_str = str(file)
        # Just because split in python can't work with more then 1 delimiter !!!
        temp_path_str = path_str.replace('.', '\\')
        # Extract file name with extension
        tokens = temp_path_str.split('\\')  # by slash
        file_name = tokens[-2]  # second from the last token is the file name
        file_name = ''.join(filter(str.isalpha, file_name))  # extract only letters
        if last_file != file_name:
            all_data.append(temp_accents_list)  # Add the list to all the data
            temp_accents_list = []
        temp_accents_list.append(path_str)
        last_file = file_name

    # first we want to remove all accents with less then 5 recording
    for list in all_data[:]:
        if len(list) < minimum_records:
            all_data.remove(list)

    print("Number of classes: ", len(all_data))
    for list in all_data:
        amount_to_training = int(len(list) * split_train_factor)
        amount_to_test = len(list) - amount_to_training  # The rest goes for test set
        # Move 80% to training
        for i in range(amount_to_training):
            source = list[i]
            tokens = source.split('\\')  # by slash
            file_name_with_extension = tokens[-1]
            destination = train_path + '\\' + file_name_with_extension
            os.rename(source, destination)  # Move the file to train_set folder
            if(i >= 85): #Maximum 86 records to train
                break
        # Move 20% to test
        for j in range(amount_to_test):
            source = list[amount_to_training + j]
            tokens = source.split('\\')  # by slash
            file_name_with_extension = tokens[-1]
            destination = test_path + '\\' + file_name_with_extension
            os.rename(source, destination)  # Move the file to test_set folder
            if(j >= 12): #Maximum 13 records to test
                break
