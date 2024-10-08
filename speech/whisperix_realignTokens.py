from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import numpy as np
import json
import os



path_audio  = '/export/b01/afavaro/tmeyer_alignment/audio_all'
path_align  = '/export/b15/tmeyer16/datasets/afavaro/tmeyer_alignment_all4'
path_output = '/export/b15/tmeyer16/datasets/afavaro/tmeyer_alignment_all4precise'
column_precise = 'start_precise'
show_plots     = False



# Iterate through all alignment files
for file in sorted( os.listdir(path_align), reverse=False):
    if file.endswith('.json'):
        print('Found: ', file)

        # Read aligment tokens
        with open(os.path.join( path_align, file), 'r') as f:
            data = json.load(f)
        data_align = pd.DataFrame.from_dict(data['word_segments'])
        if data_align.empty:
            print('  WARNING: Empty File ', file)
            continue

        # Read Raw Data
        data_fs, data_audio = wavfile.read(os.path.join( path_audio, os.path.splitext(file)[0]+'.wav'))
        time = np.linspace(0, data_audio.shape[-1]/data_fs, data_audio.shape[-1]) # sec

        # Extract and filter Envelope
        filter_envelope = 20 # Hz
        envelope = np.abs( hilbert( data_audio))
        b,a = butter(2, filter_envelope / (data_fs/2), btype='low')
        envelope = filtfilt( b, a, envelope)

        # Plot to check envelope
        if show_plots:
            plt.figure()
            plt.plot(time, data_audio, 'b', label='raw')
            plt.plot(time, envelope, 'm', label='envelope')
            plt.title(os.path.splitext(file)[0])
            plt.legend()
            plt.show()

        # Create new column for precise alignment
        data_align[column_precise] = data_align['start'].copy()
        for i, token in data_align.iterrows():
            # Get raw data for each token
            time_token = (time > token['start']) & (time < token['end'])
            token_envelope = envelope[time_token]

            # Find the "center of mass" of the token. The token starts before this "center"
            # Centroid = sum(m*x)/sum(m)
            temp = np.square(token_envelope) # square to over-emphasize large values
            token_centroid = int( np.sum( np.multiply(temp, list(range(1,1+len(temp))))) / np.sum(temp))

            # Find the last index before the centroid that is less than 10% of max
            threshold = 0.1 * (max(token_envelope) - min(token_envelope)) + min(token_envelope)
            token_possibleStart = token_envelope[0:token_centroid]
            delay_start = np.argwhere( token_possibleStart < threshold)
            if len(delay_start) > 0:
                delay_start = delay_start[-1][0]
            else:
                delay_start = 0

            # Plot to check new start
            if show_plots:
                plt.figure()
                plot_time = np.linspace(token['start'], token['end'], token_envelope.shape[-1])
                plt.plot(plot_time, data_audio[time_token], 'b', label='raw')
                plt.plot(plot_time,   envelope[time_token], 'm', label='envelope')
                plt.plot(plot_time[token_centroid],                        0, 'm*', markersize=16, label='centroid')
                plt.plot(plot_time[delay_start], token_envelope[delay_start], 'y*', markersize=16, label='start')
                plt.title(token['word'])
                plt.legend()
                plt.show()

            # Add delay to original start index
            data_align.loc[i, column_precise] = token['start'] + delay_start / data_fs
        
        data_align.to_csv( os.path.join( path_output, os.path.splitext(file)[0]+'.csv'))