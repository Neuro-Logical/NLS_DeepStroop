import matplotlib.pyplot as plt
from datetime import datetime
from numpy import linspace
from copy import deepcopy
import pandas as pd
import warnings
import sys
import os

try:
    import eyetracking.functions.extract as extract
    import eyetracking.functions.analyze as analyze
    import eyetracking.functions.annotation as annotation
    import eyetracking.functions.plot as fplot
except:
    import functions.extract as extract
    import functions.analyze as analyze
    import functions.annotation as annotation
    import functions.plot as fplot



def main():
    # Raw data filetree
    # path_data              = '/Users/trevor/datasets/eyelink/eyetracking'
    # path_data              = '/home/trevor-debian/Documents/datasets/data_eyetracking'
    path_data              = '/export/b15/tmeyer16/datasets/pubDeepStroop/nls_pubDeepStroop'
    # path_data              = '/export/b01/afavaro/new_nls'
    # Speech Alignment
    # path_alignmentRaw      = '/Users/trevor/datasets/eyelink/audio/tmeyer_alignment_all4precise'
    # path_alignmentRaw      = '/export/b15/afavaro/Trevor_paper/Alignment'
    path_alignmentRaw      = '/export/b15/tmeyer16/datasets/pubDeepStroop/align_pubDeepStroop'
    # Metadata to add initial labels
    path_metadata          = os.path.join( path_data, '0.metadata.csv')
    # Final Output
    # path_output            = '/Users/trevor/outputs/eyelink'
    # path_output            = '/home/trevor-debian/Documents/outputs/eyetracking_output'
    path_output            = '/export/b15/tmeyer16/outputs/eyelink_pubDeepStroop04'
    # Subsections to use
    notes_sheet            = 'tmeyer'
    hdf_key_dir            = 'tmeyer'
    hdf_key_note           = ''

    # Operations to perfrom
    reextract              = False
    reprocess              = True
    trim_trial             = False
    save_processed         = True
    save_csv               = True
    add_annotation         = False
    save_plots             = True
    show_plots             = False

    # Define all trials to analyze, and their naming in each file
    trials = {  'word-naming':       {'start_eye': 'Colors_preliminary1', 'end_eye': 'Colors_preliminaryEnd1', 'filename_audio': 'Secuence_stroop_Previous_1', 'trial_alignment': 'SecuenceStroopPrevious1'}, 
                'color-naming':      {'start_eye': 'Colors_preliminary2', 'end_eye': 'Colors_preliminaryEnd2', 'filename_audio': 'Secuence_stroop_Previous_2', 'trial_alignment': 'SecuenceStroopPrevious2'},
                'word-color-naming': {'start_eye':     'Word_Color_long', 'end_eye':    'Word_Color_long_END', 'filename_audio':                  'WordColor', 'trial_alignment':               'Wordcolor'},
                # 'cookieThief':      ['Exploration_Cookie', 'Exploration_CookieEnd',              'CookieThief'],
                # 'smoothPursuit1':      ['SmoothPur_1', 'SmoothPur_End1', 'NA'],
                # 'smoothPursuit2':      ['SmoothPur_2', 'SmoothPur_End2', 'NA'],
                # 'smoothPursuit3':      ['SmoothPur_3', 'SmoothPur_End3', 'NA'],
                # 'smoothPursuit4':      ['SmoothPur_4', 'SmoothPur_End4', 'NA'],
                # 'smoothPursuit5':      ['SmoothPur_5', 'SmoothPur_End5', 'NA'],
                # 'smoothPursuit6':      ['SmoothPur_6', 'SmoothPur_End6', 'NA'],
                # 'smoothPursuit7':      ['SmoothPur_7', 'SmoothPur_End7', 'NA']
            }

    # Define analysis thresholds
    analysis_constants = { 'closest_blink' :    50, # ms
                        'threshold_fixDist':    20, # pixels,      default=20
                        'threshold_fixVel' :    25, # deg/sec,     default=25
                        'threshold_fixAcc' :  3000, # deg/sec/sec, default=3000
                        'gaze_tolerance_x' :   100, # pixels
                        'gaze_tolerance_y' :  None # pixels
                        }

    # Initialize final output dataframes
    summary_ALL_sac = pd.DataFrame()
    summary_ALL_fix = pd.DataFrame()
    summary_ALL_blk = pd.DataFrame()
    summary_ALL_gaz = pd.DataFrame()
    summary_ALL_wrd = pd.DataFrame()
    summary_ALL     = pd.DataFrame()
    summary_status  = pd.DataFrame()
    status_template = {**{'edf':[False]}, **{t:[False] for t in trials.keys()}, **{t+'_mvmt':[False] for t in trials.keys()}, **{t+'_gaze':[False] for t in trials.keys()}, **{t+'_wordAlign':[False] for t in trials.keys()}, **{t+'_wordBegin':[False] for t in trials.keys()}, **{'ses'+str(s).zfill(2):[False] for s in range(1,4)}}
    # Create data extraction output folder
    path_output_data = os.path.join( path_output, 'data_processed')
    if not os.path.exists(path_output_data):
        os.mkdir(path_output_data)
    if len(hdf_key_note) > 0:
        hdf_key_suffix = '_' + hdf_key_note
    else:
        hdf_key_suffix = ''

    subject_metadata = pd.read_csv( path_metadata)
    
    # Iterate through each subject
    for subject in sorted(os.listdir(path_data), reverse=False):
        path_subject = os.path.join( path_data, subject)
        if os.path.isdir(path_subject):
            path_subject_eye    = os.path.join( path_subject, 'eyetracking')
            if not( os.path.exists(path_subject_eye)) or not( os.path.isdir( path_subject_eye)):
                continue
            path_subject_speech = os.path.join( path_subject, 'speech')
            path_out_subject    = os.path.join( path_output_data, subject)
            if not os.path.exists(path_out_subject):
                os.mkdir(path_out_subject)
            try:
                group = subject_metadata.loc[ subject_metadata['subject'] == subject, 'label'].values[0]
            except:
                group = 'UKN'

            # Initialize output dataframes for this subject
            summary_sac     = pd.DataFrame()
            summary_fix     = pd.DataFrame()
            summary_blk     = pd.DataFrame()
            summary_gaz     = pd.DataFrame()
            summary_wrd     = pd.DataFrame()
            summary_subject = pd.DataFrame()
            status_subject  = {}

            # Check if output was alrady extracted, or if it must be (re)processed
            path_output_processed = os.path.join( path_out_subject, subject+'_info.hdf')
            hdfstore_dir_exists = False
            if os.path.isfile(path_output_processed):
                store = pd.HDFStore(path_output_processed)
                hdfstore_dir_exists = any( [s.startswith('/'+hdf_key_dir+'/') for s in store.keys()])
                store.close()
            if reprocess or not(hdfstore_dir_exists):
                # Extract subject notes
                notes_subject = extract.get_subjectNotes(path_subject_eye, notes_sheet)
                # Find all the edf files in the folder
                for filename in sorted( os.listdir(path_subject_eye)):
                    if filename.lower().endswith('.edf'):
                        print('Subject:\t', '\t\t'.join([subject,group,filename]))
                        session      = filename.split('_')[2]
                        session_file = filename.split('_')[2] + '-' + os.path.splitext(filename)[0].split('-')[-1]
                        status_subject[session_file] = deepcopy( status_template)
                        status_subject[session_file]['edf'] = [True]
                        status_subject[session_file][session] = [True]

                        # Extract the raw data from the edf file, if necessary
                        path_raw     = os.path.join( path_subject_eye, filename)
                        path_extract = os.path.splitext( path_raw)[0] + '.hdf5'
                        if not( os.path.exists(path_extract)) or reextract:
                            path_intermediate = os.path.splitext( path_raw)[0] + '.asc'
                            if not( os.path.exists(path_intermediate)) or reextract:
                                extract.edf2asc(path_raw, path_intermediate)
                            extract.asc2hdf(path_intermediate, path_extract)

                        # Extract data from available files
                        notes_file = notes_subject.loc[ notes_subject['filename'] == filename]  if notes_subject is not None else None
                        data_eye_annotation = extract.hdf2df( path_extract, 'eyelink_annotations')
                        data_eye_samples    = extract.hdf2df( path_extract, 'eyelink_samples')
                        # messages_all = data_eye_annotation.loc[ data_eye_annotation.iloc[:,0] == 'MSG']
                        # message_options = messages_all.iloc[:,2].unique()
                        
                        str_date = '-'.join( data_eye_annotation.loc[1, 3:].dropna())
                        time_eyeFile = datetime.strptime(str_date, '%b-%d-%H:%M:%S-%Y')
                        session_date = '{:04d}-{:02d}-{:02d}'.format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day)

                        # Reference Validation row:
                        #       0        1     2           3    4   5      6     7      8     9     10     11   12      13    14    15          16    17    18
                        # idx  MSG  6917588  !CAL  VALIDATION  HV9  LR   LEFT  POOR  ERROR  3.97  avg.  17.41  max  OFFSET  2.86  deg.  -2.4,120.1  pix.  None
                        try:
                            msg_validation  = data_eye_annotation[ data_eye_annotation.loc[:,3] == 'VALIDATION']
                            index_lowAvgErr = msg_validation.loc[:,9].astype(float).idxmin()
                            eye_lowValError = msg_validation.loc[index_lowAvgErr, 6]
                        except:
                            print('\t\tno validation...using right\t', filename)
                            eye_lowValError = 'NoVal'
                        if any([ temp in filename for temp in ['AD_002_ses03', 'AD_014_ses03', 'NLS_074_ses01', 'NLS_104_ses01']]):
                            print('\t\tcorrecting to use right eye\t', filename)
                            eye_lowValError = 'right'
                        
                        # Iterate through all the trials of interest, defined in dictionary above
                        for trial, trial_messages in trials.items():
                            print('\t', trial)
                            sys.stdout.flush()

                            # Find the trial starting and ending timestamp
                            start_eye   = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages['start_eye']].iloc[:,1]
                            end_eye     = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[  'end_eye']].iloc[:,1]
                            start_audio = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTART') & (data_eye_annotation.iloc[:,6] == trial_messages['filename_audio']+'.wav')].iloc[:,1]
                            end_audio   = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTOP' ) & (data_eye_annotation.iloc[:,6] == trial_messages['filename_audio']+'.wav')].iloc[:,1]

                            if min(len(start_eye), len(end_eye)) == 0:
                                print('\t\tNo data found in edf')
                                continue
                                
                            # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
                            for index_trial, (timestamp_start, timestamp_end) in enumerate( zip( start_eye, end_eye)):
                                # Save a description to document observations with
                                description_trial = subject + '_' + session_file + '-' + str(index_trial) + '_' + trial
                                status_subject[session_file][trial] = [True]

                                # Extract the raw data
                                df_trial_raw = data_eye_samples[ (data_eye_samples['timestamp'] >= float(timestamp_start)) & (data_eye_samples['timestamp'] <= float(timestamp_end))].copy()
                                df_trial     = df_trial_raw.copy()
                                
                                timestamp_startAudio = start_audio[ (start_audio > timestamp_start) & (start_audio < timestamp_end)].values
                                timestamp_endAudio   = end_audio[   (end_audio   > timestamp_start) & (end_audio   < timestamp_end)].values
                                if (len(timestamp_startAudio) > 1) or (len(timestamp_endAudio) > 1):
                                    print('WARNING: AUDIO: Multiple start/end times found in EDF\t', description_trial)
                                    data_wordAlign = None
                                    data_audio     = None
                                    fs_audio       = None

                                else:
                                    data_wordAlign, data_audio, fs_audio = extract.get_wordAlignment( subject, session, trial_messages['trial_alignment'], path_alignmentRaw, path_rawData=path_subject_speech)

                                    # Uncomment to skip audio files without start/end times annotated in edf file
                                    if (len(timestamp_startAudio) == 0) or (len(timestamp_endAudio) == 0):
                                        print('WARNING: AUDIO: No start/end times found in EDF\t', description_trial)
                                        data_wordAlign = None
                                        data_audio     = None
                                        fs_audio       = None
                                    if (data_wordAlign is not None) and (data_wordAlign['time'].iloc[-1] > ((float(timestamp_end) - float(timestamp_start)) / 1000)):
                                        # The trial timestamps do not line up, we have most likely found the wrong trial
                                        print('WARNING: AUDIO: Start/end times in EDF do not align\t', description_trial)
                                        data_wordAlign = None
                                        data_audio     = None
                                        fs_audio       = None
                                    if data_audio is not None:
                                        duration_audio = data_audio.shape[-1] / fs_audio
                                        if duration_audio - ((float(timestamp_endAudio[0]) - float(timestamp_startAudio[0])) / 1000) > 0.1:
                                            # The trial durations are exactly the same, we have found the correct audio for this trial
                                            print('WARNING: AUDIO: Start/end duration in EDF is not the same as the raw audio\t', description_trial)
                                            data_wordAlign = None
                                            data_audio     = None
                                            fs_audio       = None

                                # Prepare the trial eye data
                                eye_lowError = eye_lowValError.lower()
                                if eye_lowError == 'left':
                                    df_trial.rename(columns={ 'pos_x_left':'pos_x',  'pos_y_left':'pos_y',  'vel_x_left':'vel_x',  'vel_y_left':'vel_y'}, inplace=True)
                                elif eye_lowError == 'right':
                                    df_trial.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y'}, inplace=True)
                                else:
                                    print('\t\tno validation...using right\t', filename)
                                    eye_lowError = 'right'
                                    df_trial.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y'}, inplace=True)
                                
                                # Prepare the audio data
                                if (data_wordAlign is not None) and (len(timestamp_startAudio) > 0):
                                    try:
                                        data_wordAlign.loc[:,'time'] = (data_wordAlign.loc[:,'time'] * 1000) + float(timestamp_startAudio[0])
                                        data_wordAlign['word_prefix'] = data_wordAlign.apply(lambda x: x['word'].lower().replace('▁','')[:3] if (len(x['word']) > 1) else None, axis=1)
                                    except:
                                        print('WARNING: AUDIO: Word Token data not valid\t', description_trial)
                                        data_wordAlign = None
                                        data_audio     = None
                                        fs_audio       = None

                                # Get any notes relevant to this trial
                                if notes_file is not None:
                                    notes_trial = notes_file.loc[ notes_file['trial'] == trial].copy()  if notes_file is not None else None
                                    # Convert trial-timestamp (starts at zero for each trial) notes to global raw-data defined timestamps (defined by eyelink raw data)
                                    timestamp_trialStart = df_trial.iloc[0]['timestamp']
                                    for timestamp_update in ['timestamp_start', 'timestamp_end']:
                                        row_update = notes_trial[timestamp_update] < 1000
                                        notes_trial.loc[row_update, timestamp_update] = notes_trial.loc[row_update, timestamp_update] * 1000 + timestamp_trialStart
                                else:
                                    notes_trial = None
                                
                                if (notes_trial is None) or (len(notes_trial) == 0):
                                    print('WARNING: NOTES: No notes found\t', description_trial)
                                #############################################
                                ##        BEGIN DATA ANALYSIS STEPS        ##
                                #############################################
                                try:
                                    '''
                                        Analyze: Eye Movement
                                    '''
                                    # Measure General Eye Movement
                                    df_trial, info_saccade, info_fixation, info_blink = analyze.get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y', notes=notes_trial)
                                    df_trial, info_saccade, info_fixation, info_blink = annotation.remove_eyeMovement( notes_trial, df_trial, info_saccade, info_fixation, info_blink, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)


                                    '''
                                        Analyze: Eye Gaze
                                    '''
                                    # Measure Eye gaze characterisics
                                    df_trial, info_saccade, info_fixation, info_blink, info_gaze, cluster_fcn, cluster_desc = analyze.get_eyeGazeStimuli(df_trial, info_saccade, info_fixation, info_blink, trial, 'timestamp', 'pos_x',  'pos_y', 'pos_x', 'pos_y', 'duration', col_saccade='saccade', col_fixation='fixation',  col_blink='blink', trim_trial=False, save_trimPlot=False, path_trimPlot=os.path.join( path_subject_eye,'cropTrial',description_trial+'.png'), notes=notes_trial)
                                    if cluster_fcn is None:
                                        print('WARNING: GAZE: No gaze stimuli found\t', description_trial)
                                    # df_trial, info_saccade, info_fixation, info_blink, info_gaze = annotation.update_eyeGazeStimuli( notes_trial, df_trial, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'timestamp_start', 'timestamp_end')
                                    

                                    if add_annotation:
                                        print('NOTICE: ADDING ANNOTATION\t', description_trial)
                                        notes_subject = annotation.add_annotation( notes_subject, df_trial, 'timestamp', 'pos_x', 'pos_y', 'saccade', 'blink', filename, trial, desc_suffix=eye_lowError)
                                        extract.update_subjectNotes(path_subject_eye, notes_sheet, notes_subject)

                                    '''
                                        Data Check: Decide if data analysis is good enough.
                                                    If not, try the other eye
                                    '''
                                    # Decide if we should try the other eye
                                    trial_length = (df_trial.iloc[-1]['timestamp'] - df_trial.iloc[0]['timestamp'])
                                    blink_length = info_blink['duration'].sum()
                                    missing_perc_threshold = 0.2
                                    if (blink_length > (trial_length*missing_perc_threshold)) or (info_gaze is None):
                                        print('\t\tTrying other eye...')
                                        df_trial_t = df_trial_raw.copy()
                                        # Switch Eyes
                                        if eye_lowError == 'left':
                                            eye_lowError = 'right'
                                            df_trial_t.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y'}, inplace=True)
                                        elif eye_lowError == 'right':
                                            eye_lowError = 'left'
                                            df_trial_t.rename(columns={ 'pos_x_left':'pos_x',  'pos_y_left':'pos_y',  'vel_x_left':'vel_x',  'vel_y_left':'vel_y'}, inplace=True)

                                        # Measure General Eye Movement
                                        df_trial_t, info_saccade_t, info_fixation_t, info_blink_t = analyze.get_eyeMovement(df_trial_t, analysis_constants, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y', notes=notes_trial)
                                        df_trial_t, info_saccade_t, info_fixation_t, info_blink_t = annotation.remove_eyeMovement( notes_trial, df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)
                                        # Measure Eye gaze characterisics
                                        df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, info_gaze_t, cluster_fcn_t, cluster_desc_t = analyze.get_eyeGazeStimuli(df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, trial, 'timestamp', 'pos_x',  'pos_y', 'pos_x', 'pos_y', 'duration', col_saccade='saccade', col_fixation='fixation',  col_blink='blink', trim_trial=trim_trial, save_trimPlot=save_plots, path_trimPlot=os.path.join( path_subject_eye,'trimTrial',description_trial+'.png'), notes=notes_trial)
                                        if cluster_fcn_t is None:
                                            print('WARNING: GAZE: No gaze stimuli found\t', description_trial)
                                        # df_trial_t, info_saccade, info_fixation, info_blink, info_gaze = annotation.update_eyeGazeStimuli( notes_trial, df_trial_t, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'timestamp_start', 'timestamp_end')
                                        
                                        if add_annotation:
                                            print('NOTICE: ADDING OTHER EYE ANNOTATION\t', description_trial)
                                            notes_subject = annotation.add_annotation( notes_subject, df_trial_t, 'timestamp', 'pos_x', 'pos_y', 'saccade', 'blink', filename, trial, desc_suffix=eye_lowError)
                                            extract.update_subjectNotes(path_subject_eye, notes_sheet, notes_subject)

                                        # Check if this eye was better. If so, replace all of the data so far.
                                        if ((info_gaze is None) and (info_gaze_t is not None)) or (info_blink_t['duration'].sum() < (blink_length)):
                                            df_trial      = df_trial_t
                                            info_saccade  = info_saccade_t
                                            info_fixation = info_fixation_t
                                            info_blink    = info_blink_t
                                            info_gaze     = info_gaze_t
                                            cluster_fcn   = cluster_fcn_t
                                            cluster_desc  = cluster_desc_t
                                        else:
                                            if info_gaze is None:
                                                print('WARNING: GAZE: Could not identify gaze\t', description_trial)
                                            else:
                                                print('WARNING: GAZE: Could not find enough data', description_trial)
                                    
                                    '''
                                        Analyze: Combine audio characteristics with eye characteristics
                                    '''
                                    info_wordBegin = analyze.get_wordCorrectSequenceInfo( data_wordAlign, trial, 'time', 'word_prefix', df_fixation=info_fixation, col_fixationTime='timestamp_start', col_focus='focus', col_fixationGazeIndex='gaze_index')
                                    if info_gaze is not None:
                                        info_wordBegin = analyze.get_multimodalTiming( info_wordBegin, info_fixation, 'time', 'word_index', 'alignment', 'gaze_index', 'timestamp_start', 'timestamp_end', 'focus', 'duration')
                                        if (info_wordBegin is not None) and ('time_lookBeforeWord' in info_wordBegin.columns):
                                            status_subject[session_file][trial+'_wordAlign'] = [True]
                                    valid_wordBegin = info_wordBegin.loc[~info_wordBegin['word_index'].isnull(),:].copy() if (info_wordBegin is not None) else None

                                    if (info_wordBegin is not None) and (len(info_wordBegin[ info_wordBegin['word_index'] > 0]) == 0):
                                        print('WARNING: AUDIO: No correct tokens identified for ', description_trial)


                                except Exception as e:
                                    print('ERROR processing ', description_trial)
                                    print(e)
                                    continue

                                # Track Data Validity
                                status_subject[session_file][trial+'_mvmt']      = [False] if info_saccade is None else [True]
                                status_subject[session_file][trial+'_gaze']      = [False] if info_gaze is None else [True]
                                status_subject[session_file][trial+'_wordBegin'] = [False] if info_wordBegin is None else [True]

                                # Compile summary statistics for the trial.
                                summary = {}
                                list_df_outputs = []
                                durationTrial = (df_trial.iloc[-1]['timestamp'] - df_trial.iloc[0]['timestamp']) / 1000
                                for df_info, desc in zip( [info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
                                    summary.update( analyze.get_summaryStats( df_info, durationTrial, prefix=desc, timestamp_trialStart=df_trial.iloc[0]['timestamp']))

                                    if df_info is not None:
                                        df_info['trial']         = trial
                                        df_info['session']       = session
                                        df_info['trial_index']   = session_file + '-' + str(index_trial)
                                        df_info['filename']      = filename
                                        df_info['subject']       = subject
                                        df_info['group']         = group
                                        df_info['date_recorded'] = "{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}".format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day, time_eyeFile.hour, time_eyeFile.minute, time_eyeFile.second)
                                    
                                    list_df_outputs.append(df_info)

                                [info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin] = list_df_outputs
                                
                                summary.update( analyze.get_trialStats( df_trial, info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin, 'timestamp', 'focus', 'gaze_line', 'gaze_line_start', 'gaze_line_end', 'gaze_word', 'gaze_word_start', 'gaze_word_end', 'timestamp_start', 'timestamp_end', 'correct', 'time'))
                                summary['trial']         = trial
                                summary['session']       = session
                                summary['trial_index']   = session_file + '-' + str(index_trial)
                                summary['subject']       = subject
                                summary['group']         = group
                                summary['date_recorded'] = "{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}".format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day, time_eyeFile.hour, time_eyeFile.minute, time_eyeFile.second)
                                summary['procSuccess_mvmt'] = False if info_saccade is None else True
                                summary['procSuccess_gaze'] = False if info_gaze is None else True
                                summary['procSuccess_word'] = False if info_wordBegin is None else True
                                # Add the subject output to the subject output dataframe
                                summary_sac     = summary_sac.append(  info_saccade, sort=False)
                                summary_fix     = summary_fix.append( info_fixation, sort=False)
                                summary_blk     = summary_blk.append(    info_blink, sort=False)
                                summary_subject = summary_subject.append( pd.DataFrame(summary, index=[0]), sort=False)
                                if info_gaze is not None:
                                    summary_gaz = summary_gaz.append( info_gaze, sort=False)
                                if info_wordBegin is not None:
                                    summary_wrd = summary_wrd.append( info_wordBegin, sort=False)

                                # Generate the processing plot output, if necessary
                                if show_plots or save_plots:
                                    try:
                                        folder_output = 'plot_eyeMovement'
                                        if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                            os.mkdir(os.path.join( path_out_subject, folder_output))
                                        path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                        fplot.eyePos_prepPlot(      df_trial, 'timestamp', 'pos_x', 'pos_y')
                                        fplot.plot_raw(             df_trial, 'timestamp', 'pos_x', 'pos_y', save_plots=save_plots, save_path=path_output_plots)
                                        fplot.plot_saccades(        df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade',          analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], save_plots=save_plots, save_path=path_output_plots)
                                        fplot.plot_saccades_blinks( df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade', 'blink', analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], save_plots=save_plots, save_path=path_output_plots)

                                        if info_gaze is not None:
                                            folder_output = 'plot_eyeGaze'
                                            if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                os.mkdir(os.path.join( path_out_subject, folder_output))
                                            path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                            fplot.plot_word(            df_trial, 'timestamp', 'pos_x', 'pos_y', 'fixation', 'focus', 'gaze_line', 'gaze_word', save_plots=save_plots, save_path=path_output_plots)
                                            fplot.plot_fixationStimuli( cluster_fcn, info_fixation, 'pos_x', 'pos_y', 'gaze_word', 'gaze_line', cluster_descriptions=cluster_desc, annotate=False, save_plots=save_plots, save_path=path_output_plots)
                                            fplot.plot_boundaryStimuli( cluster_fcn, cluster_descriptions=cluster_desc, annotate=True, save_plots=save_plots, save_path=path_output_plots)

                                        if info_wordBegin is not None:
                                            folder_output = 'plot_wordBegin'
                                            if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                os.mkdir(os.path.join( path_out_subject, folder_output))
                                            path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                            fplot.timestamp_prepPlot( info_fixation, ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                            fplot.timestamp_prepPlot( info_blink,    ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                            fplot.timestamp_prepPlot( info_saccade,  ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                            fplot.timestamp_prepPlot( info_wordBegin,  'time', timestamp_zero=timestamp_start)
                                            fplot.timestamp_prepPlot( valid_wordBegin, 'time', timestamp_zero=timestamp_start)

                                            fplot.plot_wordTokens_correct( info_wordBegin, 'time', 'word', 'word_index', save_plots=save_plots, save_path=path_output_plots)
                                            if info_gaze is not None:
                                                fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'time', 'time_lookBeforeWord', col_color='word_prefix', save_plots=save_plots, save_path=path_output_plots)
                                                fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'time', 'time_lookAfterWord',  col_color='word_prefix', save_plots=save_plots, save_path=path_output_plots)
                                                fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'time', 'time_lookRatio',      col_color='word_prefix', save_plots=save_plots, save_path=path_output_plots)
                                                fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'time', 'time_lookDuration',   col_color='word_prefix', save_plots=save_plots, save_path=path_output_plots)
                                                if (data_audio is not None) and (fs_audio is not None):
                                                    time_start_audio = (float(timestamp_startAudio) - float(timestamp_start)) / 1000
                                                    time_audio = linspace(time_start_audio, time_start_audio+(data_audio.shape[0]/fs_audio), data_audio.shape[0])
                                                    fplot.plot_alignedGazeAudio ( info_saccade, info_fixation, info_blink, info_wordBegin, data_audio, time_audio, 'time', title=trial, save_plots=save_plots, save_path=path_output_plots)


                                    
                                    except Exception as e:
                                        print('ERROR plotting ', description_trial)
                                        print(e)
                                        # Close all figures to preserve memory
                                        for fig in plt.get_fignums():
                                            plt.figure( fig)
                                            plt.clf()
                                            plt.close()
                                        continue

                                    if show_plots:
                                        plt.show()
                                    # Close all figures to preserve memory
                                    for fig in plt.get_fignums():
                                        plt.figure( fig)
                                        plt.clf()
                                        plt.close()

                                if eye_lowError == 'left':
                                    df_trial.rename(columns={'pos_x':'pos_x_left',  'pos_y':'pos_y_left',  'vel_x':'vel_x_left',  'vel_y':'vel_y_left'},  inplace=True)
                                elif eye_lowError == 'right':
                                    df_trial.rename(columns={'pos_x':'pos_x_right', 'pos_y':'pos_y_right', 'vel_x':'vel_x_right', 'vel_y':'vel_y_right'}, inplace=True)


                # After extracting all trials from all present files, save the data to .hdf file
                print('\n\tSaving Results...')

                if save_processed:
                    for df in [summary_sac, summary_fix, summary_blk, summary_subject, summary_gaz, summary_wrd]:
                        if df is None:
                            continue
                        for col in reversed(['subject', 'group', 'session', 'trial', 'trial_index', 'date_recorded']):
                            if col in df.columns:
                                df.insert( 0, col, df.pop(col))
                                # col_move = df.pop(col)
                                # df = pd.concat([col_move, df], axis=1)
                        df = df.copy()

                    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
                    # If existing extracted data exists, remove it
                    store = pd.HDFStore(path_output_processed)
                    if hdf_key_dir in store.keys():
                        store.remove(hdf_key_dir)
                    store.close()

                    summary_sac.to_hdf(  path_output_processed, hdf_key_dir+'/saccade'+hdf_key_suffix,  mode='a')
                    summary_fix.to_hdf( path_output_processed, hdf_key_dir+'/fixation'+hdf_key_suffix, mode='a')
                    summary_blk.to_hdf(    path_output_processed, hdf_key_dir+'/blink'+hdf_key_suffix,    mode='a')
                    summary_subject.to_hdf(  path_output_processed, hdf_key_dir+'/summary'+hdf_key_suffix,  mode='a')
                    if len(summary_gaz) > 0:
                        summary_gaz.to_hdf( path_output_processed, hdf_key_dir+'/gaze'+hdf_key_suffix,     mode='a')
                    if len(summary_wrd) > 0:
                        summary_wrd.to_hdf( path_output_processed, hdf_key_dir+'/wordBegin'+hdf_key_suffix, mode='a')
                    
                    if save_csv:
                        path_output_processed_csv = os.path.join(path_out_subject, 'output_csv')
                        if not(os.path.exists(path_output_processed_csv)) or not(os.path.isdir(path_output_processed_csv)):
                            os.mkdir(path_output_processed_csv)
                        summary_sac.to_csv(  os.path.join(path_output_processed_csv, subject+'_saccade_'  +hdf_key_dir+hdf_key_suffix+'.csv'))
                        summary_fix.to_csv( os.path.join(path_output_processed_csv, subject+'_fixation_' +hdf_key_dir+hdf_key_suffix+'.csv'))
                        summary_blk.to_csv(    os.path.join(path_output_processed_csv, subject+'_blink_'    +hdf_key_dir+hdf_key_suffix+'.csv'))
                        summary_subject.to_csv(  os.path.join(path_output_processed_csv, subject+'_summary_'  +hdf_key_dir+hdf_key_suffix+'.csv'))
                        if len(summary_gaz) > 0:
                            summary_gaz.to_csv( os.path.join(path_output_processed_csv, subject+'_gaze_'      +hdf_key_dir+hdf_key_suffix+'.csv'))
                        if len(summary_wrd) > 0:
                            summary_wrd.to_csv( os.path.join(path_output_processed_csv, subject+'_wordBegin_'+hdf_key_dir+hdf_key_suffix+'.csv'))

            # If path_output_processed already exists, simply import the existing data (do not re-process)
            else:
                print('Loading ', subject, '...')
                summary_sac  = pd.read_hdf( path_output_processed, hdf_key_dir+'/saccade'+hdf_key_suffix)
                summary_fix = pd.read_hdf( path_output_processed, hdf_key_dir+'/fixation'+hdf_key_suffix)
                summary_blk    = pd.read_hdf( path_output_processed, hdf_key_dir+'/blink'+hdf_key_suffix)
                summary_subject  = pd.read_hdf( path_output_processed, hdf_key_dir+'/summary'+hdf_key_suffix)
                status_subject['imported']['edf']         = [True]
                status_subject['imported'][trial+'_mvmt'] = [True]
                try:
                    summary_gaz = pd.read_hdf( path_output_processed, hdf_key_dir+'/gaze'+hdf_key_suffix)
                    status_subject['imported'][trial+'_gaze'] = [True]
                except:
                    print('WARNING: GAZE: No gaze data found for ', subject)
                    summary_gaz = pd.DataFrame()
                try:
                    summary_wrd = pd.read_hdf( path_output_processed, hdf_key_dir+'/wordBegin'+hdf_key_suffix)
                    status_subject['imported'][trial+'_wordBegin'] = [True]
                except:
                    print('WARNING: AUDIO: No word data found for ', subject)
                    summary_wrd = pd.DataFrame()

                if 'group' in summary_sac.columns:
                    summary_sac.loc[ :,'group'] = group
                    summary_fix.loc[:,'group'] = group
                    summary_blk.loc[   :,'group'] = group
                    summary_subject.loc[ :,'group'] = group
                    if len(summary_gaz) > 0:
                        summary_gaz.loc[    :,'group'] = group
                    if len(summary_wrd) > 0:
                        summary_wrd.loc[    :,'group'] = group

            # Append this subjects data to the overall summary array
            summary_ALL_sac = summary_ALL_sac.append(summary_sac,   sort=False)
            summary_ALL_fix = summary_ALL_fix.append(summary_fix,  sort=False)
            summary_ALL_blk = summary_ALL_blk.append(summary_blk,     sort=False)
            summary_ALL_gaz = summary_ALL_gaz.append(summary_gaz,      sort=False)
            summary_ALL_wrd = summary_ALL_wrd.append(summary_wrd, sort=False)
            summary_ALL = summary_ALL.append(summary_subject,   sort=False)
            for session, status in status_subject.items():
                summary_status = summary_status.append(pd.DataFrame({**{'subject':subject,'group':group}, **status}), sort=False)


    path_output_all = os.path.join( path_output, 'data_summary.hdf')
    summary_ALL_sac.to_hdf( path_output_all, hdf_key_dir+'/saccade'+hdf_key_suffix,   mode='a')
    summary_ALL_fix.to_hdf( path_output_all, hdf_key_dir+'/fixation'+hdf_key_suffix,  mode='a')
    summary_ALL_blk.to_hdf( path_output_all, hdf_key_dir+'/blink'+hdf_key_suffix,     mode='a')
    summary_ALL_gaz.to_hdf( path_output_all, hdf_key_dir+'/gaze'+hdf_key_suffix,      mode='a')
    summary_ALL_wrd.to_hdf( path_output_all, hdf_key_dir+'/wordBegin'+hdf_key_suffix, mode='a')
    summary_ALL.to_hdf( path_output_all, hdf_key_dir+'/summary'+hdf_key_suffix,   mode='a')
    for col in reversed(['subject', 'edf', 'trial', 'trial_index', 'date_recorded']):
        if col in summary_status.columns:
            summary_status.insert( 0, col, summary_status.pop(col))
            summary_status = summary_status.copy()
    summary_status.replace({True:1,False:0}).to_csv(os.path.join( path_output, 'data_status.csv'))



if __name__ == '__main__':
    main()
    print('\n\nFin.')
