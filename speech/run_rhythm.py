from scipy.stats import kurtosis, skew
import pandas as pd
import numpy as np
import os



path_align  = '/export/b15/tmeyer16/datasets/pubDeepStroop/align_pubDeepStroop'
path_output = '/export/b15/tmeyer16/outputs/eyelink_pubDeepStroop02/speech_processed'



all_files = [os.path.join(path_align, elem) for elem in os.listdir(path_align)]
audio_files = []
std_rhyt = []
kurt_rhyt = []
skew_rhyt = []
for file in all_files:
    read_file = pd.read_csv(file)
    audio_files.append(os.path.basename(file).split(".csv")[0])
    std_rhyt.append(np.std(read_file['start_precise'].tolist()))
    skew_rhyt.append(skew(read_file['start_precise'].tolist()))
    kurt_rhyt.append(kurtosis(read_file['start_precise'].tolist()))

dict = {'audio_files': audio_files, 'std_rhyt': std_rhyt, 'kurt_rhyt': kurt_rhyt, 'skew_rhyt': skew_rhyt}
df   = pd.DataFrame(dict)
df.to_csv(os.path.join(path_output, 'rhythm.csv'))