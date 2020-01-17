from pathlib import Path

from pydub import AudioSegment

'''In order to install ffmpeg, follow the 4 steps in:
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/
'''
if __name__ == "__main__":
    path = 'C:/Users/Dekel/Downloads/לימודים/deep_learning/datasets/speech-accent-archive/recordings'
    path_list = Path(path).glob('**/*.mp3')

    for file in path_list:  # file is a full path to mp3 file
        # because path is object not string
        path_str = str(file)
        # Just because split in python can't work with more then 1 delimiter !!!
        temp_path_str = path_str.replace('.', '\\')
        # Extract file name with extension
        tokens = temp_path_str.split('\\')  # by slash
        file_name = tokens[-2]  # second from the last token is the file name
        sound = AudioSegment.from_mp3(path + "/" + file_name + ".mp3")
        sound.export(path + "/wav/" + file_name + ".wav", format="wav")
