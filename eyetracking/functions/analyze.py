from copy import deepcopy
import pandas as pd
import numpy as np
import os

try:
    import eyetracking.functions.process as process
    import eyetracking.functions.annotation as annotation
except:
    try:
        import functions.process as process
        import functions.annotation as annotation
    except:
        import process as process
        import annotation as annotation



def get_eyeMovement( df_trial, process_values, col_timestamp, col_pos_x, col_pos_y, col_vel_x=None, col_vel_y=None, data_label=None, label_description=None, notes=None):
    '''Takes a dataframe of raw eye position data and extracts saccade, fixation, and blink data.

    If velocity data does not exist, it is calculated from the position data. This conversion requires a resolution column input
    to designate a specific pixels to degrees conversion for each respective screen location to convert pixels/sec to degrees/sec.

    If acceleration data does not exist, it is calculated from the velocity data.

    Args:
        df_trial (pandas dataframe):        Dataframe containing all the raw data for a particular trial to analyze.
        process_values (dictionary):        Dictionary of processing values/thresholds to pass to :py:func:`~eyetracking.functions.process.get_saccades` and :py:func:`~eyetracking.functions.process.get_blinks`
        col_timestamp (str):                Column header where timestamps (ms) for each data sample.
        col_pos_x (str):                    Column header where eye position (pixels) in the x position can be found.
        col_pos_y (str):                    Column header where eye position (pixels) in the y position can be found.
        col_vel_x (str, optional):          Column header where eye velocity (deg/s) in the x position can be found.
        col_vel_y (str, optional):          Column header where eye velocity (deg/s) in the y position can be found.
        data_label (str, optional):         Label to apply to the passed data. This will be appended to column names and added to the output dataframes.
                                            Common use is to designate data as LEFT/RIGHT, BEST, AVG, etc. to differentiate different data sources.
        label_description (str, optional):  Describes the label passed in the ``data_label`` parameter.
        notes (pandas dataframe, optional): Dataframe containing notes about the data for that trial.  Currently not implemented.
    
    Returns:
        df_trial (pandas dataframe, contains original data along with eye movement data columns added), 
        info_saccade (pandas dataframe, contains information about identified saccades provided by :py:func:`~eyetracking.functions.analyze.get_saccadeStats`), 
        info_fixation (pandas dataframe, contains information about identified fixations provided by :py:func:`~eyetracking.functions.analyze.get_fixationStats`), 
        info_blink (pandas dataframe, contains information about identified blinks provided by :py:func:`~eyetracking.functions.analyze.get_blinkStats`)
    '''
    # Copy the trial to avoid changing the source unexpectedly
    df_trial = df_trial.copy()

    # Append the data label to the column headers
    if data_label is None:
        data_suffix = ''
    elif not data_label.startswith('_'):
        data_suffix = '_' + data_label
    else:
        data_suffix = data_label
    col_vel      = 'vel'      + data_suffix
    col_acel_x   = 'acel_x'   + data_suffix
    col_acel_y   = 'acel_y'   + data_suffix
    col_acel     = 'acel'     + data_suffix
    col_blink    = 'blink'    + data_suffix
    col_saccade  = 'saccade'  + data_suffix
    col_fixation = 'fixation' + data_suffix

    # Calculate velocity
    time_diff = df_trial[col_timestamp].diff() / 1e3 # Convert to seconds
    if any( [c is None for c in [col_vel_x, col_vel_y]]) \
     or any( [(c not in df_trial.columns) for c in [col_vel_x, col_vel_y]]):
        col_vel_x = 'vel_x' + data_suffix
        col_vel_y = 'vel_y' + data_suffix
        df_trial[col_vel_x] = df_trial[col_pos_x].diff().divide(time_diff)
        df_trial[col_vel_y] = df_trial[col_pos_y].diff().divide(time_diff)
    df_trial[col_vel] = (df_trial[col_vel_x].pow(2) + df_trial[col_vel_y].pow(2)).pow(0.5)
    
    # Calculate acceleration
    df_trial[col_acel_x] = df_trial[col_vel_x].diff().divide(time_diff)
    df_trial[col_acel_y] = df_trial[col_vel_y].diff().divide(time_diff)
    df_trial[col_acel]   = (df_trial[col_acel_x].pow(2) + df_trial[col_acel_y].pow(2)).pow(0.5)

    # Identify blinks
    df_trial[col_blink]    = process.get_blinks(   df_trial, [col_pos_x, col_pos_y], col_vel, col_acel, closest_blink=process_values['closest_blink'], threshold_fixVel=process_values['threshold_fixVel'], threshold_fixAcc=process_values['threshold_fixAcc'])
    # Identify saccades
    df_trial[col_saccade]  = process.get_saccades( df_trial,  col_pos_x, col_pos_y,  col_vel, col_acel, col_blink, threshold_fixDist=process_values['threshold_fixDist'], threshold_fixVel=process_values['threshold_fixVel'], threshold_fixAcc=process_values['threshold_fixAcc'])
    # Identify fixations
    df_trial[col_fixation] = (~df_trial[col_saccade] & ~df_trial[col_blink])

    # Add the data label to each output dataframe
    if data_label is None:
        data_label = 'one'
    if label_description is None:
        label_description = 'eye'

    # Extract info from identified saccades
    saccade      = process.get_indexStartStop( df_trial[col_saccade])
    info_saccade = []
    for i in range(len(saccade)):
        data_saccade = df_trial.loc[saccade[i][0]:saccade[i][1]].reset_index(inplace=False, drop=True)
        info_saccade.append( get_saccadeStats(data_saccade, col_timestamp, col_pos_x,  col_pos_y, col_vel))
    if len(info_saccade) > 0:
        info_saccade = pd.DataFrame(info_saccade)
    else:
        info_saccade = pd.DataFrame(columns=get_saccadeStats(df_trial,col_timestamp,col_pos_x,col_pos_y,col_vel).keys())
    info_saccade[label_description] = data_label

    # Extract info from identified fixation
    fixation      = process.get_indexStartStop( df_trial[col_fixation])
    info_fixation = []
    for i in range(len(fixation)):
        data_fixation = df_trial.loc[fixation[i][0]:fixation[i][1]].reset_index(inplace=False, drop=True)
        info_fixation.append( get_fixationStats(data_fixation, col_timestamp, col_pos_x, col_pos_y, col_blink))
    if len(info_fixation) > 0:
        info_fixation = pd.DataFrame(info_fixation)
    else:
        info_fixation = pd.DataFrame(columns=get_fixationStats(df_trial,col_timestamp,col_pos_x,col_pos_y,col_blink).keys())
    info_fixation[label_description] = data_label

    # Extract info from identified blink
    blink    = process.get_indexStartStop( df_trial[col_blink])
    info_blink = []
    for i in range(len(blink)):
        data_blink = df_trial.loc[blink[i][0]:blink[i][1]].reset_index(inplace=False, drop=True)
        info_blink.append( get_blinkStats(data_blink, col_timestamp, col_pos_x, col_pos_y))
    if len(info_blink) > 0:
        info_blink = pd.DataFrame(info_blink)
    else:
        info_blink = pd.DataFrame(columns=get_blinkStats(df_trial,col_timestamp,col_pos_x,col_pos_y).keys())
    info_blink[label_description] = data_label


    return df_trial, info_saccade, info_fixation, info_blink



def get_eyeGazeStimuli( df_trial, df_saccade, df_fixation, df_blink, name_stimuli, col_timestamp, col_pos_x, col_pos_y, col_fix_pos_x, col_fix_pos_y, col_fix_dur, col_saccade=None, col_fixation=None, col_blink=None, trim_trial=True, save_trimPlot=False, path_trimPlot=None, data_label=None, label_description=None, notes=None):
    '''Takes a dataframe of raw data and corresponding extracting saccade, fixation, and blink data and extracts information about which stimuli the subject is gazing at.

    Also returns a cluster predictor function to classify other fixation points in other datasets.

    Args:
        df_trial (pandas dataframe):        Dataframe containing all the raw data for a particular trial to analyze.
        df_saccade (pandas dataframe):      Dataframe containing information about identified saccades provided by :py:func:`~eyetracking.functions.analyze.get_saccadeStats`
        df_fixation (pandas dataframe):     Dataframe containing information about identified fixations provided by :py:func:`~eyetracking.functions.analyze.get_fixationStats`
        df_blink (pandas dataframe):        Dataframe containing information about identified blinks provided by :py:func:`~eyetracking.functions.analyze.get_blinkStats`
        name_stimuli (str):                 Name of the target stimuli to use when identifying fixations. String must be present in :py:func:`~eyetracking.functions.process.get_stimuliDescriptions`
        col_timestamp (str):                Column header where timestamps (ms) for each data sample.
        col_pos_x (str):                    Column header in `df_trial` where eye position (pixels) in the x position can be found.
        col_pos_y (str):                    Column header in `df_trial` where eye position (pixels) in the y position can be found.
        col_fix_pos_x (str):                Column header in `df_fixation` where eye position (pixels) in the x position can be found.
        col_fix_pos_y (str):                Column header in `df_fixation` where eye position (pixels) in the y position can be found.
        col_fix_dur (str):                  Column header in `df_fixation` where fixation duration (milliseconds) can be found.
        col_saccade (str, optional):        Column header in `df_trial` where saccade status can be found. Only used to avoid trimming trial mid-saccade.
        col_fixation (str, optional):       Column header in `df_trial` where fixation status can be found. Is not currently used.
        col_blink (str, optional):          Column header in `df_trial` where blink status can be found. Only used to avoid trimming trial mid-blink.
        trim_trial (bool, optional):        Whether to trim the trial using :py:func:`~eyetracking.functions.process.trim_wordTrial`
        save_trimPlot (bool, optional):     Whether to save a plotted summary of the trim_trial 
        path_trimPlot (str, optional):      Path to save the output plotted summary of trim_trial
        data_label (str, optional):         Label to apply to the passed data. This will be appended to column names and added to the output dataframes. Common use is to designate data as LEFT/RIGHT, BEST, AVG, etc. to differentiate different data sources.
        label_description (str, optional):  Describes the label passed in the ``data_label`` parameter.
        notes (pandas dataframe, optional): Dataframe containing notes about the data for that trial.  Currently not implemented.

    Returns:
        df_trial (pandas dataframe, contains original data along with gaze stimuli data columns added), 
        df_saccade (pandas dataframe, contains original data along with gaze stimuli data columns added), 
        df_fixation (pandas dataframe, contains original data along with gaze stimuli data columns added), 
        df_blink (pandas dataframe, contains original data along with gaze stimuli data columns added),
        info_gaze (pandas dataframe, contains information about identified saccades provided by :py:func:`~eyetracking.functions.analyze.get_gazeStats`)
        cluster_fcn (sklearn nearest neighbor predictor (n_neighbors=1) fit to the extracted word centroids. Provided by :py:func:`~eyetracking.functions.process.get_stimuliFixation`, could be used for prediction)
        stimuli_desc (dict, characteristics of each stimuli. Provided by :py:func:`~eyetracking.functions.process.get_stimuliDescriptions`)
    '''
    df_trial = df_trial.copy()

    if data_label is None:
        data_suffix = ''
    elif not data_label.startswith('_'):
        data_suffix = '_' + data_label
    else:
        data_suffix = data_label
    col_gaze       = 'gaze'       + data_suffix
    col_gaze_line  = 'gaze_line'  + data_suffix
    col_gaze_word  = 'gaze_word'  + data_suffix
    col_gaze_color = 'gaze_color' + data_suffix
    col_gaze_text  = 'gaze_text'  + data_suffix
    col_gaze_index = 'gaze_index' + data_suffix
    col_focus      = 'focus'      + data_suffix

    col_time_start = col_timestamp + '_start' + data_suffix
    col_time_end   = col_timestamp + '_end'   + data_suffix

    # Identify the stimuli
    df_fixation, cluster_fcn, (num_words, num_lines) = process.get_stimuliFixation(df_fixation, name_stimuli, col_time_start, col_time_end, col_fix_pos_x, col_fix_pos_y, col_fix_dur, col_gaze=col_gaze, col_gaze_line=col_gaze_line, col_gaze_word=col_gaze_word, col_gaze_text=col_gaze_text, col_gaze_color=col_gaze_color, col_focus=col_focus, notes=notes)
    df_fixation = annotation.corect_gazeStimuli( notes, df_fixation, col_time_start, col_time_end, col_gaze_line, col_gaze_word)
    df_fixation, stimuli_desc = process.get_stimuliFixation_desc(df_fixation, name_stimuli, num_words, num_lines, col_gaze_line, col_gaze_word, col_gaze_text, col_gaze_color, col_gaze, col_gaze_index)

    if cluster_fcn is None:
        # Something failed
        return df_trial, df_saccade, df_fixation, df_blink, None, cluster_fcn, None

    ## Trim the trial based on the stimuli
    if trim_trial:
        if save_trimPlot:
            df_trial_untrimmed = df_trial.copy()
        trial_start, trial_end, df_trial, [df_saccade, df_fixation, df_blink] = process.trim_wordTrial( df_trial, col_timestamp, col_time_start, col_time_end, col_gaze_line, col_gaze_word, events_avoid=[col_saccade, col_blink], df_summary_trim=[df_saccade, df_fixation, df_blink])
        if save_trimPlot:
            import functions.plot as fplot
            fplot.plot_trim(df_trial_untrimmed, trial_start, trial_end, col_timestamp, col_pos_x, col_pos_y, save_plots=save_trimPlot, save_path=path_trimPlot)
            del df_trial_untrimmed
    
    df_saccade = process.add_eventStimuli(df_saccade, df_fixation, col_time_start, col_time_end, col_gaze, col_gaze_line, col_gaze_word, col_gaze_color, col_gaze_text, col_gaze_index, col_focus)
    df_blink   = process.add_eventStimuli(df_blink,   df_fixation, col_time_start, col_time_end, col_gaze, col_gaze_line, col_gaze_word, col_gaze_color, col_gaze_text, col_gaze_index, col_focus)
    df_trial   = process.add_trialStimuli(df_trial,   df_fixation, [df_saccade, df_blink], col_timestamp, col_time_start, col_time_end, col_gaze, col_gaze_line, col_gaze_word, col_gaze_text, col_gaze_color, col_gaze_index, col_focus)
    
    if data_label is None:
        data_label = 'one'
    if label_description is None:
        label_description = 'eye'
    
    # Extract info from identified gaze
    info_gaze = []
    for stimuli in sorted( df_fixation[col_gaze].unique()):
        # Skip the first and last lines
        stimuli_parts = stimuli.split(',')
        # if (stimuli_parts[1] == '0') or (stimuli_parts[1] == str(num_lines-1)) or (int(stimuli_parts[0]) < 0) or (int(stimuli_parts[1]) < 0):
        if (int(stimuli_parts[0]) < 0) or (int(stimuli_parts[1]) < 0):
            continue
        saccade_stimuli  = df_saccade[ (df_saccade['gaze_start'] == stimuli) | (df_saccade['gaze_end'] == stimuli)]
        fixation_stimuli = df_fixation[ df_fixation[col_gaze] == stimuli ]
        blink_stimuli     = df_blink[ (df_blink['gaze_start'] == stimuli) & (df_blink['gaze_end'] == stimuli)]

        # Decided not to analyze saccade/fixation/blink per word, to avoid injecting too much information
        info_gaze.append( get_gazeStats( stimuli, saccade_stimuli, fixation_stimuli, blink_stimuli, 'pos_x', 'pos_start_x', 'pos_end_x', 'pos_y', 'pos_start_y', 'pos_end_y', 'gaze_start', 'gaze_end', 'duration', 'distance', 'velocity_avg', 'velocity_max', singleStimuli=True))

    info_gaze = pd.DataFrame(info_gaze)
    info_gaze[label_description] = data_label

    return df_trial, df_saccade, df_fixation, df_blink, info_gaze, cluster_fcn, stimuli_desc



def get_wordCorrectSequenceInfo( df_wordBegin, trial, col_time, col_token, df_fixation=None, col_fixationTime=None, col_focus=None, col_fixationGazeIndex=None):
    '''Determines whether the word tokens which are present are confident to be correct. Logic rests mainly in :py:func:`~eyetracking.functions.process.get_wordCorrectSequence`

    Args:
        df_wordBegin (pandas dataframe):  Dataframe containing information about the begining time of each token
        trial (str):                      Trial of interest. Used to determine expected vocabulary from  :py:func:`~eyetracking.functions.process.get_tokenVocabulary`
        col_time (str):                   Column header where timestamps for each token can be found.
        col_token (str):                  Column header where tokens can be found
        df_fixation (pandas dataframe, optional):  Dataframe containing information about each fixation from eyetracking
        col_fixationTime (str, optional): Column header where fixation times can be found
        col_focus (str, optional):        Column header where focus indication can be found
        col_fixationGazeIndex (str, optional): Column header where gaze index can be found

    Returns:
        df_wordBegin with correct word indexes designated.
    '''
    if df_wordBegin is None:
        return None
    
    col_inVocab   = 'inVocab'
    col_delay     = 'delay'
    col_repeat    = 'repeat'
    col_correct   = 'correct'
    col_align     = 'alignment'
    col_wordIndex = 'word_index'
    col_wordLine  = 'word_line'
    col_wordWord  = 'word_word'

    # Clean out of vocabulary words
    task_vocabulary = process.get_tokenVocabulary(trial)

    if len(task_vocabulary) > 0:
        df_wordBegin[col_inVocab] = df_wordBegin[col_token].isin(task_vocabulary.flatten())
    else:
        df_wordBegin[col_inVocab] = True

    repeated_minTime = 0 # ms # Set to 0 bc this check no longer needed with newest speech transcription/alignment software
    df_wordClean = process.clean_wordTokens( df_wordBegin, col_token, col_repeat, col_time, col_delay, col_inVocab, repeated_minTime=repeated_minTime)
    if df_wordClean.empty:
        print('WARNING: AUDIO: No correct tokens found')
        df_wordBegin[col_correct]   = False
        df_wordBegin[col_wordIndex] = np.nan
        df_wordBegin[col_wordLine]  = np.nan
        df_wordBegin[col_wordWord]  = np.nan
        return df_wordBegin

    # Get the "truth" stimuli descriptions
    stimuli = process.get_stimuliDescriptions(trial)

    # Get whether the word sequence is correct
    if (df_fixation is not None) and (col_fixationGazeIndex in df_fixation.columns): # and len(df_fixation.loc[:,col_fixationGazeIndex].max() > 0)
        df_wordClean = process.get_wordCorrectSequence( df_wordClean, df_fixation, stimuli, col_time, col_token, col_fixationTime, col_focus, col_fixationGazeIndex, col_wordIndex)
    else:
        df_wordClean = process.get_wordCorrectSequence_onlyAudio( df_wordClean, stimuli, col_token, col_wordIndex)
    df_wordBegin[col_wordIndex] = df_wordClean[col_wordIndex]
    df_wordBegin[col_correct]   = df_wordClean[col_wordIndex].notna()
    df_wordBegin[col_correct]   = df_wordBegin[col_correct].fillna(False)
    if col_align in df_wordClean.columns:
        df_wordBegin[col_align] = df_wordClean[col_align]
    else:
        df_wordBegin[col_align] = False

    # find indexes for correct words
    clusters_initial_x, clusters_initial_y = process.get_clusterLocations(trial)
    num_words = len(clusters_initial_x)
    num_lines = len(clusters_initial_y)
    df_wordBegin[col_wordLine] = df_wordBegin[col_wordIndex] // num_words
    df_wordBegin[col_wordWord] = df_wordBegin[col_wordIndex]  % num_words

    return df_wordBegin



def get_multimodalTiming( df_wordBegin, df_fixation, col_time, col_audWordIndex, col_goodAlign, col_fixWordIndex, col_fixTimeStart, col_fixTimeEnd, col_fixFocus, col_fixDuration):
    '''Gets the precise timing of eye fixation on stimuli compared to when the correct stimuli is spoken.

    Args:
        df_wordBegin (pandas dataframe): Dataframe containing information about spoken tokens
        df_fixation (pandas dataframe):  Dataframe containing information about individual saccades
        col_time (str):                  Column header in `df_wordBegin` where timestamps for each token can be found.
        col_audWordIndex (str):          Column header in `df_wordBegin` where the spoken word index can be found
        col_goodAlign (str):             Column header in `df_wordBegin` where the alignment verification can be found
        col_fixWordIndex (str):          Column header in `df_fixation` where the gaze word index can be found
        col_fixTimeStart (str):          Column header in `df_fixation` where the start timestamp can be found
        col_fixTimeEnd (str):            Column header in `df_fixation` where the end timestamp can be found
        col_fixFocus (str):              Column header in `df_fixation` where a designation of focus can be found
        col_fixDuration (str):           Column header in `df_fixation` where the fixation duraiton can be found

    Returns:
        df_wordBegin with the multimodal timings added.
    '''
    if df_wordBegin is None:
        return None
    
    if df_wordBegin[col_goodAlign].mean() < 0.66:
        return df_wordBegin # If alignments are too bad, do not continue
    else:
        valid_wordBegin = df_wordBegin.loc[((df_wordBegin[col_goodAlign]) & (~df_wordBegin[col_audWordIndex].isnull())),:].copy()
    
    col_lookBefore   = 'time_lookBeforeWord'
    col_lookAhead    = 'time_lookAfterWord'
    col_lookRatio    = 'time_lookRatio'
    col_lookDuration = 'time_lookDuration'

    # Initialize Columns
    df_wordBegin[col_lookBefore]   = np.nan
    df_wordBegin[col_lookAhead]    = np.nan
    df_wordBegin[col_lookRatio]    = np.nan
    df_wordBegin[col_lookDuration] = np.nan

    # Calculate some statistics for all of the valid words
    for i, row in valid_wordBegin.iterrows():
        try:
            time_beforeWord = row[col_time] - min( df_fixation[ (df_fixation[col_fixWordIndex] == row[col_audWordIndex]) & (df_fixation[col_fixFocus])][col_fixTimeStart])
        except:
            time_beforeWord = 0
        try:
            time_afterWord  = max( df_fixation[ (df_fixation[col_fixWordIndex] == row[col_audWordIndex]) & (df_fixation[col_fixFocus])][col_fixTimeEnd]) - row[col_time]
        except:
            time_afterWord  = 0
        duration_gaze = sum( df_fixation[ (df_fixation[col_fixWordIndex] == row[col_audWordIndex]) & (df_fixation[col_fixFocus])][col_fixDuration])
        df_wordBegin.loc[i,col_lookBefore]   = time_beforeWord * 1e-3
        df_wordBegin.loc[i,col_lookAhead]    = time_afterWord  * 1e-3
        if time_afterWord == 0:
            df_wordBegin.loc[i,col_lookRatio]    = np.nan
        else:
            df_wordBegin.loc[i,col_lookRatio]    = time_beforeWord / time_afterWord
        df_wordBegin.loc[i,col_lookDuration] = duration_gaze   * 1e-3
    
    return df_wordBegin



def get_saccadeStats(df_saccade, col_timestamp, col_pos_x, col_pos_y, col_vel, timestamp_trialStart=0):
    '''Extracts explicit characteristics of each saccade, and compiles them into one dataframe.
    This is often useful for saccade characterization and statistical analysis.

    Args:
        df_saccade (pandas dataframe): Dataframe containing raw data corresponding to a single saccade.
        col_timestamp (str):           Column header where timestamps (ms) for each data sample.
        col_pos_x (str):               Column header in `df_saccade` where eye position (pixels) in the x position can be found.
        col_pos_y (str):               Column header in `df_saccade` where eye position (pixels) in the y position can be found.
        col_vel (str):                 Column header in `df_saccade` where eye speed (deg/s) can be found.
        timestamp_trialStart (float, optional): Offset correction for timestamp stats, if desired.
    
    Returns:
        stats_sac (dict, containing all extracted saccade characteristics)
    '''
    stats_sac = {}
    stats_sac['timestamp_start'] = df_saccade.iloc[ 0][col_timestamp] - timestamp_trialStart
    stats_sac['timestamp_end']   = df_saccade.iloc[-1][col_timestamp] - timestamp_trialStart
    stats_sac['pos_start_x']     = df_saccade.iloc[ 0][col_pos_x]
    stats_sac['pos_start_y']     = df_saccade.iloc[ 0][col_pos_y]
    stats_sac['pos_end_x']       = df_saccade.iloc[-1][col_pos_x]
    stats_sac['pos_end_y']       = df_saccade.iloc[-1][col_pos_y]
    distance_all                 = np.sqrt(np.square(df_saccade.iloc[0][col_pos_x] - df_saccade[col_pos_x]) + np.square(df_saccade.iloc[0][col_pos_y] - df_saccade[col_pos_y]))
    stats_sac['distance']        = distance_all.iloc[-1]
    stats_sac['overshoot']       = max(distance_all) - stats_sac['distance']
    stats_sac['duration']        = stats_sac['timestamp_end'] - stats_sac['timestamp_start']
    stats_sac['velocity_avg']    = np.mean( df_saccade[col_vel].abs())
    stats_sac['velocity_max']    = np.amax( df_saccade[col_vel].abs())

    return stats_sac


def get_fixationStats(df_fixation, col_timestamp, col_pos_x, col_pos_y, col_blink, timestamp_trialStart=0):
    '''Extracts explicit characteristics of each fixation, and compiles them into one dataframe.
    This is often useful for fixation characterization and statistical analysis.

    Args:
        df_fixation (pandas dataframe): Dataframe containing raw data corresponding to a single fixation.
        col_timestamp (str):            Column header where timestamps (ms) for each data sample.
        col_pos_x (str):                Column header in `df_fixation` where eye position (pixels) in the x position can be found.
        col_pos_y (str):                Column header in `df_fixation` where eye position (pixels) in the y position can be found.
        col_blink (str):                Column header in `df_fixation` where eye speed (deg/s) can be found.
        timestamp_trialStart (float, optional): Offset correction for timestamp stats, if desired.
    
    Returns:
        stats_fix (dict, containing all extracted fixation characteristics)
    '''
    stats_fix = {}
    stats_fix['timestamp_start'] = df_fixation.iloc[ 0][col_timestamp] - timestamp_trialStart
    stats_fix['timestamp_end']   = df_fixation.iloc[-1][col_timestamp] - timestamp_trialStart
    stats_fix['pos_x']           = df_fixation[col_pos_x].mean()
    stats_fix['pos_y']           = df_fixation[col_pos_y].mean()
    stats_fix['duration']        = stats_fix['timestamp_end'] - stats_fix['timestamp_start']
    blinks                       = process.get_indexStartStop(df_fixation[col_blink])
    stats_fix['num_blinks']      = len( blinks)
    if stats_fix['num_blinks']:
        stats_fix['blink_delay'] = df_fixation.iloc[blinks[0][0]][col_timestamp] - df_fixation.iloc[0][col_timestamp]
    else:
        stats_fix['blink_delay'] = np.NaN
    
    return stats_fix


def get_blinkStats(df_blink, col_timestamp, col_pos_x, col_pos_y, timestamp_trialStart=0):
    '''Extracts explicit characteristics of each blink, and compiles them into one dataframe.
    This is often useful for blink characterization and statistical analysis.

    Args:
        df_blink (pandas dataframe): Dataframe containing raw data corresponding to a single blink.
        col_timestamp (str):         Column header where timestamps (ms) for each data sample.
        col_pos_x (str):             Column header in `df_blink` where eye position (pixels) in the x position can be found.
        col_pos_y (str):             Column header in `df_blink` where eye position (pixels) in the y position can be found.
        timestamp_trialStart (float, optional): Offset correction for timestamp stats, if desired.
        
    Returns:
        stats_fix (dict, containing all extracted blink characteristics)
    '''
    stats_blk = {}
    stats_blk['timestamp_start'] = df_blink.iloc[ 0][col_timestamp] - timestamp_trialStart
    stats_blk['timestamp_end']   = df_blink.iloc[-1][col_timestamp] - timestamp_trialStart
    stats_blk['pos_start_x']     = df_blink.iloc[ 0][col_pos_x]
    stats_blk['pos_start_y']     = df_blink.iloc[ 0][col_pos_y]
    stats_blk['pos_end_x']       = df_blink.iloc[-1][col_pos_x]
    stats_blk['pos_end_y']       = df_blink.iloc[-1][col_pos_y]
    stats_blk['duration']        = df_blink.iloc[-1][col_timestamp] - df_blink.iloc[0][col_timestamp]
    stats_blk['distance']        = np.sqrt(np.square(df_blink.iloc[0][col_pos_x] - df_blink.iloc[-1][col_pos_x]) + np.square(df_blink.iloc[0][col_pos_y] - df_blink.iloc[-1][col_pos_y]))

    return stats_blk


def get_gazeStats( stimuli, df_saccade, df_fixation, df_blink, col_pos_x, col_pos_start_x, col_pos_end_x, col_pos_y, col_pos_start_y, col_pos_end_y, col_gaze_start, col_gaze_end, col_duration, col_distance, col_velocityAvg, col_velocityMax, singleStimuli=False):
    '''Extracts explicit characteristics of an individual target stimuli to be gazed at, and compiles them into one dataframe.
    This is often useful for gaze characterization for each stimuli in a trial and statistical analysis.

    Args:
        stimuli (entry in dataframe):   Dataframe entry to look for in col_gaze_start to identify the stimuli of interest
        df_saccade (pandas dataframe):  Dataframe containing saccade data corresponding to a single target gaze stimuli.
        df_fixation (pandas dataframe): Dataframe containing fixation data corresponding to a single target gaze stimuli.
        df_blink (pandas dataframe):    Dataframe containing blink data corresponding to a single target gaze stimuli.
        col_pos_x (str):                Column header in `df_fixation` where eye position (pixels) in the x position can be found.
        col_pos_start_x (str):          Column header in `df_saccade` where starting eye position (pixels) in the x position can be found.
        col_pos_end_x (str):            Column header in `df_saccade` where ending eye position (pixels) in the x position can be found.
        col_pos_y (str):                Column header in `df_fixation` where eye position (pixels) in the y position can be found.
        col_pos_start_y (str):          Column header in `df_saccade` where starting eye position (pixels) in the y position can be found.
        col_pos_end_y (str):            Column header in `df_saccade` where ending eye position (pixels) in the y position can be found.
        col_gaze_start (str):           Column header in `df_saccade` and `df_blink` where gaze stimuli at the start of the event can be found.
        col_gaze_end (str):             Column header in `df_saccade` and `df_blink` where gaze stimuli at the end of the event can be found.
        col_duration (str):             Column header in `df_fixation` and `df_blink` where the event duration can be found
        col_distance (str):             Column header in `df_saccade` where the saccade distance can be found.
        col_velocityAvg (str):          Column header in `df_saccade` where the average velocity can be found.
        col_velocityMax (str):          Column header in `df_saccade` where the maximum velocity can be found.
        singleStimuli (bool, optional): Whether the past data is represents a gaze at a single stimuli.
        
    Returns:
        stats_gaz (dict, containing all extracted gaze characteristics)
    '''
    stats_gaz = {}
    if len(df_saccade) == 0:
        return stats_gaz
    
    stats_gaz['stimuli'] = stimuli
    if len(stimuli.split(',')) > 1:
        stats_gaz['gaze_word'] = stimuli.split(',')[0]
        stats_gaz['gaze_line'] = stimuli.split(',')[1]

    stats_gaz['num_fixations']     = len(df_fixation)
    stats_gaz['duration_sacTotal'] = df_saccade[ col_duration].sum()
    stats_gaz['duration_fixTotal'] = df_fixation[col_duration].sum()
    stats_gaz['duration_blkTotal'] = df_blink[   col_duration].sum()
    stats_gaz['duration_total']    = df_saccade[ col_duration].sum() + df_fixation[col_duration].sum() + df_blink[col_duration].sum()
    stats_gaz['duration_sacAvg']   = df_saccade[ col_duration].mean()
    stats_gaz['duration_fixAvg']   = df_fixation[col_duration].mean()
    stats_gaz['duration_blkAvg']   = df_blink[   col_duration].mean()
    
    stats_gaz['pos_variance']    = np.sqrt( np.square( df_fixation[col_pos_x].std()) + np.square( df_fixation[col_pos_y].std()))

    stats_gaz['num_sac']  = len(df_saccade)
    stats_gaz['num_sacForward']  = ((df_saccade[col_pos_end_x] - df_saccade[col_pos_start_x]) > 0).sum()
    stats_gaz['num_sacBackward'] = ((df_saccade[col_pos_end_x] - df_saccade[col_pos_start_x]) < 0).sum()
    stats_gaz['num_sacUp']   = ((df_saccade[col_pos_end_y] - df_saccade[col_pos_start_y]) > 0).sum()
    stats_gaz['num_sacDown'] = ((df_saccade[col_pos_end_y] - df_saccade[col_pos_start_y]) < 0).sum()

    stats_gaz['distance_sac']  = (df_saccade[ col_pos_end_x] - df_saccade[ col_pos_start_x]).sum()
    sac_forward = df_saccade[((df_saccade[col_pos_end_x] - df_saccade[col_pos_start_x]) > 0)]
    stats_gaz['distance_sacForward']  = (sac_forward[ col_pos_end_x] - sac_forward[ col_pos_start_x]).sum()
    sac_backward = df_saccade[((df_saccade[col_pos_end_x] - df_saccade[col_pos_start_x]) < 0)]
    stats_gaz['distance_sacBackward'] = (sac_backward[col_pos_end_x] - sac_backward[col_pos_start_x]).sum()
    stats_gaz['distance_sacHorizontal'] = ( df_saccade[col_pos_end_x] - df_saccade[col_pos_start_x]).abs().sum()
    stats_gaz['distance_sacVertical']   = ( df_saccade[col_pos_end_y] - df_saccade[col_pos_start_y]).abs().sum()

    saccade_internal = df_saccade.loc[ df_saccade['gaze_same']]
    stats_gaz['num_sacInternal']              = saccade_internal.shape[0]
    stats_gaz['distance_sacInternal']         = saccade_internal[col_distance].sum()
    stats_gaz['velocityAvg_sacInternalMean']  = saccade_internal[col_velocityAvg].mean()
    stats_gaz['velocityMax_sacInternalMean']  = saccade_internal[col_velocityMax].mean()
    stats_gaz['num_sacInternalForward']       = ((saccade_internal[col_pos_end_x] - saccade_internal[col_pos_start_x]) > 0).sum()
    stats_gaz['num_sacInternalbackward']      = ((saccade_internal[col_pos_end_x] - saccade_internal[col_pos_start_x]) < 0).sum()
    stats_gaz['num_sacInternalUp']            = ((saccade_internal[col_pos_end_y] - saccade_internal[col_pos_start_y]) > 0).sum()
    stats_gaz['num_sacInternalDown']          = ((saccade_internal[col_pos_end_y] - saccade_internal[col_pos_start_y]) < 0).sum()

    sac_backwardInt = saccade_internal[((saccade_internal[col_pos_end_x] - saccade_internal[col_pos_start_x]) > 0)]
    sac_forwardInt  = saccade_internal[((saccade_internal[col_pos_end_x] - saccade_internal[col_pos_start_x]) < 0)]
    stats_gaz['distance_sacInternalbackward'] = (sac_backwardInt[col_pos_end_x] - sac_backwardInt[col_pos_start_x]).sum()
    stats_gaz['distance_sacInternalforward']  = (sac_forwardInt[ col_pos_end_x] - sac_forwardInt[ col_pos_start_x]).sum()
    stats_gaz['distance_sacInternalHorizontal'] = (saccade_internal[col_pos_end_x] - saccade_internal[col_pos_start_x]).abs().sum()
    stats_gaz['distance_sacInternalVertical']   = (saccade_internal[col_pos_end_y] - saccade_internal[col_pos_start_y]).abs().sum()

    if singleStimuli:
        saccade_entry = df_saccade.loc[ df_saccade[col_gaze_start] != stimuli]
        if saccade_entry.shape[0] > 0:
            stats_gaz['distance_sacEntryFirst']    = saccade_entry[col_distance].iloc[0]
            stats_gaz['velocityAvg_sacEntryFirst'] = saccade_entry[col_velocityAvg].iloc[0]
            stats_gaz['velocityMax_sacEntryFirst'] = saccade_entry[col_velocityMax].iloc[0]

        saccade_exit = df_saccade.loc[ df_saccade[col_gaze_end  ] != stimuli]
        if saccade_exit.shape[0] > 0:
            stats_gaz['distance_sacExitLast']    = saccade_exit[col_distance].iloc[-1]
            stats_gaz['velocityAvg_sacExitLast'] = saccade_exit[col_velocityAvg].iloc[-1]
            stats_gaz['velocityMax_sacExitLast'] = saccade_exit[col_velocityMax].iloc[-1]


    stats_gaz['num_blinks'] = len(df_blink)

    if singleStimuli:
        stats_gaz['num_visits'] = (df_saccade[col_gaze_start] != stimuli).sum() + (df_blink[col_gaze_start] != stimuli).sum()

    # Potential metrics to extract
    # Error rates/order rate
    # Number leave/revisit
    # Order of progression

    return stats_gaz


def get_summaryStats( df_metric, durationTrial, prefix='', timestamp_trialStart=0):
    '''Extracts summary characteristics of all columns in a passed dataframe.
    Summary characteristics attempt to summarize the distribution/occurances of each metric using a standardized summary method.
    Often useful if trying to summarize characteristics accross subjects/trials as this standardizes the summary of each metric.

    If the metric is numeric, the mean, median, standar deviation, 10th percentile, and 90th percentile will be calculated, if possible.

    If the metric is not numeric, the mode of the metric is calculated

    The number of occurances of all metrics will be extracted.

    Note:
        If a column labeled 'eye' is found in the dataframe and has more than one unique entry, individual summarization characteristics will be calculated for each eye.
        
        If a column labeled 'eye' does not exist or only has one entry, only one set of summarization characteristics will be extracted.

    Args:
        df_metric (pandas dataframe): Dataframe containing saccade data corresponding to a single target gaze stimuli.
        durationTrial (float, optional): The duration of the trial, used to calculate
        prefix (str, optional):          
        timestamp_trialStart (float, optional): Offset correction for timestamp stats, if desired.

    Returns:
        summary (dict, containing all extracted metric occurance characteristics)
    '''
    summary = {}
    if df_metric is None:
        return summary

    def _get_stats( dict_summary, df_data, metric, label, durationTrial, prefix='', timestamp_trialStart=0):
        if df_data[metric].isna().all():
            return dict_summary

        if len(prefix) > 0:
            label_metric = '_'.join([prefix, metric, label])
        else:
            label_metric = '_'.join( metric, label)

        if pd.api.types.is_numeric_dtype( df_data[metric]):
            if ('timestamp' in metric):
                data_use = df_data[metric] - timestamp_trialStart
            else:
                data_use = df_data[metric]
            dict_summary[label_metric+'_mean']         = data_use.mean(skipna=True)
            dict_summary[label_metric+'_median']       = data_use.median(skipna=True)
            dict_summary[label_metric+'_stdev']        = data_use.std(skipna=True)
            try:
                dict_summary[label_metric+'_perc-10'] = np.nanpercentile( data_use.to_numpy(), 10)
                dict_summary[label_metric+'_perc-90'] = np.nanpercentile( data_use.to_numpy(), 90)
            except:
                pass
            dict_summary[label_metric+'_perTimeTrial'] = data_use.sum(skipna=True) / durationTrial
        else:
            dict_summary[label_metric+'_mode'] = df_data[metric].mode()[0]
        
        return dict_summary


    if ('eye' not in df_metric.columns) or (len(df_metric['eye'].unique()) == 1):
        # Get one summary for the only eye
        for metric in df_metric.columns:
            summary = _get_stats( summary, df_metric, metric, 'one', durationTrial, prefix=prefix, timestamp_trialStart=timestamp_trialStart)
        summary[prefix+'_one_count'] = df_metric.shape[0]
        summary[prefix+'_one_rate']  = df_metric.shape[0] / durationTrial

    elif len(df_metric['eye'].unique()) > 1:
        # First, Get summary for both eyes combined
        for metric in df_metric.columns:
            summary = _get_stats( summary, df_metric, metric, 'both', durationTrial, prefix=prefix, timestamp_trialStart=timestamp_trialStart)
        summary[prefix+'_both_count'] = df_metric.shape[0]
        summary[prefix+'_both_rate']  = df_metric.shape[0] / durationTrial

        # Then, Get summary for each eye, separately
        for eye in df_metric['eye'].unique():
            df_info_eye = df_metric.loc[ df_metric['eye'] == eye]
            for metric in df_info_eye.columns:
                if metric == 'eye':
                    continue
                summary = _get_stats( summary, df_info_eye, metric, str(eye), durationTrial, prefix=prefix, timestamp_trialStart=timestamp_trialStart)
            summary[prefix+'_'+str(eye)+'_count'] = df_info_eye.shape[0]
            summary[prefix+'_'+str(eye)+'_rate']  = df_metric.shape[0] / durationTrial
        
        summary[prefix+'_eye'] = 'both'

    return summary


def get_trialStats( df_trial, df_saccade, df_fixation, df_blink, df_gaze, df_word, col_timestamp, col_focus, col_gaze_line, col_gaze_line_start, col_gaze_line_end, col_gaze_word, col_gaze_word_start, col_gaze_word_end, col_timestamp_start, col_timestamp_end, col_wordCorrect, col_wordTime, stratify_metrics=True):
    '''Extracts characteristics from the entire trial to be used for analysis

    Args:
        df_trial (pandas dataframe):    Dataframe containing raw data corresponding to an entire trial.
        df_saccade (pandas dataframe):  Dataframe containing saccade data corresponding to an entire trial.
        df_fixation (pandas dataframe): Dataframe containing fixation data corresponding to an entire trial.
        df_blink (pandas dataframe):    Dataframe containing blink data corresponding to an entire trial.
        df_gaze (pandas dataframe):     Dataframe containing gaze data corresponding to an entire trial.
        df_word (pandas dataframe):     Dataframe containing word token begining times corresponding to an entire trial.
        col_timestamp (str):            Column header in `df_trial` where timestamps (ms) for each data sample.
        col_focus (str):                Column header in `df_fixation` where the focus indicator can be found.
        col_gaze_line (str):            Column header in `df_gaze` where the line index being gazed at exists.
        col_gaze_line_start (str):      Column header in `df_saccade` where the starting line index being gazed at exists.
        col_gaze_line_end (str):        Column header in `df_saccade` where the ending line index being gazed at exists.
        col_gaze_word (str):            Column header in `df_gaze` where the word index being gazed at exists.
        col_gaze_word_start (str):      Column header in `df_saccade` where the starting word index being gazed at exists.
        col_gaze_word_end (str):        Column header in `df_saccade` where the ending word index being gazed at exists.
        col_timestamp_start (str):      Column header in `df_fixation` where timestamps (ms) for the fixation start exists.
        col_timestamp_end (str):        Column header in `df_fixation` where timestamps (ms) for the fixation end exists.
        col_wordCorrect (str)           Column header in `df_word` where timestams for word correctness exists.
        col_wordTime (str):             Column header in `df_word` where timestamps for the word beginning exists.
        stratify_metrics (bool, optional): Whether to also calculate metrics on a per-line or per-word basis.

    Returns:
        summary (dict, containing all extracted metric occurance characteristics)
    '''
    summary = {}

    # First, get some general-use variables about the trial
    timestamp_startTrial = 0
    if df_trial is not None:
        timestamp_startTrial = df_trial[col_timestamp].iloc[0]
        timestamp_endTrial   = df_trial[col_timestamp].iloc[-1]
    else:
        if df_saccade is not None:
            timestamp_startSaccade = df_saccade[col_timestamp_start].min()
            timestamp_endSaccade   = df_saccade[col_timestamp_end].max()
        if df_fixation is not None:
            timestamp_startFixation = df_fixation[col_timestamp_start].min()
            timestamp_endFixation   = df_fixation[col_timestamp_end  ].max()
            if df_saccade is None:
                timestamp_startTrial = timestamp_startFixation
                timestamp_endTrial   = timestamp_endFixation
            else:
                timestamp_startTrial = min( timestamp_startSaccade, timestamp_startFixation)
                timestamp_endTrial   = max( timestamp_endSaccade,   timestamp_endFixation)


    # Extract statistics
    summary['trial_length'] = (timestamp_endTrial - timestamp_startTrial) / 1000

    if (df_gaze is not None) and (df_gaze.shape[0] > 0):
        timestamp_startFocus = df_fixation.loc[df_fixation[col_focus], col_timestamp_start].min()
        timestamp_endFocus   = df_fixation.loc[df_fixation[col_focus], col_timestamp_start].max()

        summary['focus_length'] = (timestamp_endFocus - timestamp_startFocus) / 1000

        summary['time_beforeFocus']         = (timestamp_startFocus - timestamp_startTrial) / 1000
        summary['numSaccades_beforeFocus']  = df_saccade[  df_saccade[ col_timestamp_start] < timestamp_startFocus].shape[0]
        summary['numFixations_beforeFocus'] = df_fixation[ df_fixation[col_timestamp_start] < timestamp_startFocus].shape[0]
        summary['numBlinks_beforeFocus']    = df_blink[    df_blink[   col_timestamp_start] < timestamp_startFocus].shape[0]
        
        summary['time_afterFocus']          = (timestamp_endTrial - timestamp_endFocus) / 1000
        summary['numSaccades_afterFocus']   = df_saccade[   df_saccade[  col_timestamp_end] > timestamp_endFocus].shape[0]
        summary['numFixations_afterFocus']  = df_fixation[  df_fixation[ col_timestamp_end] > timestamp_endFocus].shape[0]
        summary['numBlinks_afterFocus']     = df_blink[     df_blink[    col_timestamp_end] > timestamp_endFocus].shape[0]

        if stratify_metrics:
            saccade_focus  = df_saccade[   df_saccade[col_focus]]
            fixation_focus = df_fixation[ df_fixation[col_focus]]
            blink_focus    = df_blink[       df_blink[col_focus]]

            summary['time_until_focus'] = (fixation_focus.loc[:,col_timestamp_start].min() - timestamp_startTrial) / 1000
            
            duration_focus = (fixation_focus[col_timestamp_end].max()  - fixation_focus[col_timestamp_start].min()) / 1000
            for df_info, desc in zip( [saccade_focus, fixation_focus, blink_focus], ['sac', 'fix', 'blk']):
                summary.update( get_summaryStats( df_info, duration_focus, prefix='focus_'+desc, timestamp_trialStart=fixation_focus[col_timestamp_start].min()))

            for line in df_fixation[col_gaze_line].unique():
                if int(line) < 0:
                    continue
                saccade_line  = df_saccade[ (df_saccade[col_gaze_line_start].astype(float) == line) | (df_saccade[col_gaze_line_end].astype(float) == line)]
                fixation_line = df_fixation[ df_fixation[col_gaze_line].astype(float) == line ]
                blink_line    = df_blink[ (df_blink[col_gaze_line_start].astype(float) == line) & (df_blink[col_gaze_line_end].astype(float) == line)]
                gaze_line     = df_gaze[ df_gaze[col_gaze_line].astype(float) == line ]

                summary['time_until_line_' + str(line)] = (fixation_line.loc[:,col_timestamp_start].min() - timestamp_startTrial) / 1000
                
                duration_line = (fixation_line[col_timestamp_end].max()  - fixation_line[col_timestamp_start].min()) / 1000
                for df_info, desc in zip( [saccade_line, fixation_line, blink_line, gaze_line], ['sac', 'fix', 'blk', 'gaz']):
                # for df_info, desc in zip( [saccade_line, fixation_line, blink_line, gaze_line, info_wordBegin], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
                    summary.update( get_summaryStats( df_info, duration_line, prefix='line-'+str(line)+'_'+desc, timestamp_trialStart=fixation_line[col_timestamp_start].min()))

                # #  Below calculates metric with each line as a new stimuli. Different from above with averages multiple stimuli interactions within one line.
                # # Only uncomment if confident this is what you want, can make metric nomenclature confusing
                # line_gazStats = get_gazeStats( 'newStimuliLine-'+str(line), saccade_line, fixation_line, blink_line, 'pos_x', 'pos_start_x', 'pos_end_x', 'pos_y', 'pos_start_y', 'pos_end_y',  'gaze_start', 'gaze_end', 'duration', 'distance', 'velocity_avg', 'velocity_max', singleStimuli=False)
                # keys_gazStats = deepcopy( list( line_gazStats.keys()))
                # for key in keys_gazStats:
                #     line_gazStats[ 'newStimuliLine-'+str(line)+'_'+key] = line_gazStats.pop(key)
                # summary.update( line_gazStats)


            for word in df_fixation[col_gaze_word].unique():
                if int(word) < 0:
                    continue
                saccade_word  = df_saccade[ (df_saccade[col_gaze_word_start].astype(float) == word) | (df_saccade[col_gaze_word_end].astype(float) == word)]
                fixation_word = df_fixation[ df_fixation[col_gaze_word].astype(float) == word ]
                blink_word    = df_blink[ (df_blink[col_gaze_word_start].astype(float) == word) & (df_blink[col_gaze_word_end].astype(float) == word)]
                gaze_word     = df_gaze[ df_gaze[col_gaze_word].astype(float) == word ]

                duration_word = (fixation_word[col_timestamp_end].max()  - fixation_word[col_timestamp_start].min()) / 1000
                for df_info, desc in zip( [saccade_word, fixation_word, blink_word, gaze_word], ['sac', 'fix', 'blk', 'gaz']):
                # for df_info, desc in zip( [saccade_word, fixation_word, blink_word, gaze_word, info_wordBegin], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
                    summary.update( get_summaryStats( df_info, duration_word, prefix='word-'+str(word)+'_'+desc, timestamp_trialStart=fixation_word[col_timestamp_start].min()))

                # # Below calculates metric with each word as a new stimuli. Different from above with averages multiple stimuli interactions within one line.
                # # Only uncomment if confident this is what you want, can make metric nomenclature confusing
                # word_gazStats = get_gazeStats( 'recalcWord-'+str(word), saccade_word, fixation_word, blink_word, 'pos_x', 'pos_start_x', 'pos_end_x', 'pos_y', 'pos_start_y', 'pos_end_y',  'gaze_start', 'gaze_end', 'duration', 'distance', 'velocity_avg', 'velocity_max', singleStimuli=False)
                # keys_gazStats = deepcopy( list( word_gazStats.keys()))
                # for key in keys_gazStats:
                #     word_gazStats[ 'recalcWord-'+str(word)+'_'+key] = word_gazStats.pop(key)
                # summary.update( word_gazStats)

    if (df_word is not None) and (df_word.shape[0] > 0):
        temp_correct = pd.Series( [False] +   df_word[col_wordCorrect].to_list()  + [False])
        temp_wrong   = pd.Series( [False] + (~df_word[col_wordCorrect]).to_list() + [False])
        correct_runs = process.get_indexStartStop(temp_correct)
        wrong_runs   = process.get_indexStartStop(temp_wrong)
        # get_indexStartStop will extend one index beyond the true end. We must subtract the last index by one
        correct_runs = [[a-1,b-1] for a,b in correct_runs]
        wrong_runs   = [[a-1,b-1] for a,b in wrong_runs]

        if len(correct_runs) > 0:
            length_correct_runs = []
            time_correct_runs   = []
            for a,b in correct_runs:
                length_correct_runs.append(b-a)
                b_use = min( b, df_word.shape[0]-1)
                time_correct_runs.append((df_word[col_wordTime].iloc[b_use] - df_word[col_wordTime].iloc[a]) / 1000)
            summary['percentCorrect']        = df_word[col_wordCorrect].sum() / df_word.shape[0]
            summary['numCorrectRuns']        =    len( length_correct_runs)
            summary['correctRunLenMean']     = np.mean(length_correct_runs)
            summary['correctRunLenStdev']    = np.std( length_correct_runs)
            summary['correctRunLenShortest'] =    min( length_correct_runs)
            summary['correctRunLenLongest']  =    max( length_correct_runs)
            summary['correctRunTimeMean']     = np.mean( time_correct_runs)
            summary['correctRunTimeStdev']    = np.std(  time_correct_runs)
            summary['correctRunTimeShortest'] =    min(  time_correct_runs)
            summary['correctRunTimeLongest']  =    max(  time_correct_runs)
        else:
            summary['percentCorrect']        = 0
            summary['numCorrectRuns']        = 0
            summary['correctRunLenMean']     = 0
            summary['correctRunLenStdev']    = 0
            summary['correctRunLenShortest'] = 0
            summary['correctRunLenLongest']  = 0
            summary['correctRunTimeMean']     = 0
            summary['correctRunTimeStdev']    = 0
            summary['correctRunTimeShortest'] = 0
            summary['correctRunTimeLongest']  = 0

        if len(wrong_runs) > 0:
            length_wrong_runs = []
            time_wrong_runs   = []
            for a,b in wrong_runs:
                length_wrong_runs.append(b-a)
                b_use = min( b, df_word.shape[0]-1)
                time_wrong_runs.append((df_word[col_wordTime].iloc[b_use] - df_word[col_wordTime].iloc[a]) / 1000)
            summary['numWrongRuns']        =    len( length_wrong_runs)
            summary['wrongRunLenMean']     = np.mean(length_wrong_runs)
            summary['wrongRunLenStdev']    = np.std( length_wrong_runs)
            summary['wrongRunLenShortest'] =    min( length_wrong_runs)
            summary['wrongRunLenLongest']  =    max( length_wrong_runs)
            summary['wrongRunTimeMean']     = np.mean( time_wrong_runs)
            summary['wrongRunTimeStdev']    = np.std(  time_wrong_runs)
            summary['wrongRunTimeShortest'] =    min(  time_wrong_runs)
            summary['wrongRunTimeLongest']  =    max(  time_wrong_runs)
        else:
            summary['numWrongRuns']        = 0
            summary['wrongRunLenMean']     = 0
            summary['wrongRunLenStdev']    = 0
            summary['wrongRunLenShortest'] = 0
            summary['wrongRunLenLongest']  = 0
            summary['wrongRunTimeMean']     = 0
            summary['wrongRunTimeStdev']    = 0
            summary['wrongRunTimeShortest'] = 0
            summary['wrongRunTimeLongest']  = 0
    
    return summary



if __name__ == '__main__':

    import process as process
    import extract as extract
    import sys

    import time
    start_time = time.time()

    path_data = '/Users/trevor/Dropbox/Mac (2)/Documents/datasets/eyelink'

    # Variables
    subject = 'NLS_6'
    trials         = {'stroop': ['Word_Color_long', 'Word_Color_long_END']}
    trim_trial     = True

    analysis_constants = { 'closest_blink' :       50, # ms
                           'threshold_fixDist' :   35, # pixels,      default=20
                           'threshold_fixVel' :    30, # deg/sec,     default=25
                           'threshold_fixAcc' :  4000, # deg/sec/sec, default=3000
                           'gaze_tolerance_x' :   100, # pixels
                           'gaze_tolerance_y' :  None # pixels
                         }
    # Read in the raw data
    target_filename = subject + '.hdf5'
    path_file       = ''
    for root, dirs, files in os.walk(path_data):
        for filename in files:
            print(filename)
            if filename == target_filename:
                path_file = os.path.join(root, filename)
                break
        if len(path_file) > 0:
            break
    if not len(path_file) > 0:
        print( target_filename, ' not found.')
        sys.exit()

    data_eye_annotation = extract.hdf2df( path_file, 'eyelink_annotations')
    data_eye_samples    = extract.hdf2df( path_file, 'eyelink_samples')

    for trial, trial_messages in trials.items():
        # Extract the timestamps of interest
        start = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[0]].iloc[:,1]
        end   = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[1]].iloc[:,1]
        # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
        for index_trial, (row_start, row_end) in enumerate( zip( start, end)):
            # Save a description to document observations with
            description_trial = subject + '_' + trial + '-' + str(index_trial)

            df_trial = data_eye_samples[ (data_eye_samples['timestamp'] >= float(start)) & (data_eye_samples['timestamp'] <= float(end))]
            
            df_trial, info_saccade_L, info_fixation_L, info_blink_L = get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x_left',  'pos_y_left',  'vel_x_left', 'vel_y_left',  data_label='left')
            df_trial, info_saccade_R, info_fixation_R, info_blink_R = get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x_right', 'pos_y_right', 'vel_x_right','vel_y_right', data_label='right')
            df_trial, info_saccade_L, info_fixation_L, info_blink_L, info_gaze_L = get_eyeGazeStimuli(df_trial, info_saccade_L, info_fixation_L, info_blink_L, analysis_constants, trial, 'timestamp', 'pos_x_left',  'pos_y_left', 'pos_x', 'pos_y', col_saccade='saccade_left',  col_fixation='fixation_left',  col_blink='blink_left',  trim_trial=True, data_label='left')
            df_trial, info_saccade_R, info_fixation_R, info_blink_R, info_gaze_R = get_eyeGazeStimuli(df_trial, info_saccade_R, info_fixation_R, info_blink_R, analysis_constants, trial, 'timestamp', 'pos_x_right', 'pos_y_right','pos_x', 'pos_y', col_saccade='saccade_right', col_fixation='fixation_right', col_blink='blink_right', trim_trial=True, data_label='right')
            
            print('\ninfo_saccade')
            print(info_saccade_L)
            # print(info_saccade_R)
            print('\ninfo_fixation')
            print(info_fixation_L)
            # print(info_fixation_R)
            print('\ninfo_blink')
            print(info_blink_L)
            # print(info_blink_R)
            print('\ninfo_gaze')
            print(info_gaze_L)
            # print(info_gaze_R)
    
    print('TOTAL TIME: ', time.time() - start_time)
