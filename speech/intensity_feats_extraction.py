import sys
sys.path.append("/export/b14/afavaro/Acoustic_Features/speech-features/speech-features/")

import pandas as pd
import os
import parselmouth
from feature_extraction_utils import *

#output_dir = '/export/b01/afavaro/feats_laure/without/'
#sound_dir = '/export/b01/afavaro/data_exp_original/'
output_dir = '/export/b16/afavaro/Results_CSV/last_data_trevor/'
sound_dir = '/export/b15/tmeyer16/datasets/pubDeepStroop/speech_pubDeepStroop/'
sound_dir_files = [os.path.join(sound_dir, elem) for elem in sorted(os.listdir(sound_dir))]
print(len(sound_dir_files))

df_tot = []
for file in sound_dir_files:

 #   if 'PEC_059_ses01_Wordcolor' not in file:

        df = pd.DataFrame()
        attributes = {}
        sound = parselmouth.Sound(file)
        sound.scale_intensity(70)

        intensity_attributes = get_intensity_attributes(sound)[0]
        pitch_attributes = get_pitch_attributes(sound)[0]
        attributes.update(intensity_attributes)
        attributes.update(pitch_attributes)

        formant_attributes = get_formant_attributes(sound)[0]
        attributes.update(formant_attributes)

        #hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound)[0]
        #gne_attributes = get_glottal_to_noise_ratio_attributes(sound)[0]
        #attributes.update(hnr_attributes)
        #attributes.update(gne_attributes)
#
        #df['local_jitter'] = None
        #df['local_shimmer'] = None
        #df.at[0, 'local_jitter'] = get_local_jitter(sound)
        #df.at[0, 'local_shimmer'] = get_local_shimmer(sound)
#
        #spectrum_attributes = get_spectrum_attributes(sound)[0]
        #attributes.update(spectrum_attributes)

        formant_attributes = get_formant_attributes(sound)[0]
        attributes.update(formant_attributes)
#
   #     lfcc_matrix, mfcc_matrix = get_lfcc(sound), get_mfcc(sound)
   #     df['lfcc'] = None
   #     df['mfcc'] = None
   #     df.at[0, 'lfcc'] = lfcc_matrix
   #     df.at[0, 'mfcc'] = mfcc_matrix
#
   #     delta_mfcc_matrix = get_delta(mfcc_matrix)
   #     delta_delta_mfcc_matrix = get_delta(delta_mfcc_matrix)
   #     df['delta_mfcc'] = None
   #     df['delta_delta_mfcc'] = None
   #     df.at[0, 'delta_mfcc'] = delta_mfcc_matrix
   #     df.at[0, 'delta_delta_mfcc'] = delta_delta_mfcc_matrix
#
        for attribute in attributes:
            df.at[0, attribute] = attributes[attribute]

        df.at[0, 'sound_filepath'] = file
        rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
        df = df[rearranged_columns]
        df_tot.append(df)

new_df = pd.concat(df_tot)
out_path = os.path.join(output_dir, "intensity_attributes.csv")
new_df.to_csv(out_path)