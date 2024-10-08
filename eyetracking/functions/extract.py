from scipy.io import wavfile
import pandas as pd
import subprocess
import warnings
import json
import sys
import os


def edf2asc( path_file, path_output):
    '''Converts and eyelink data file (.edf) to an ascii file (.asc).

    Important:
        EyeLink Developers Kit - Command Line Interface must be installed on the machine running this function.
        Information can be found here: https://www.sr-support.com/thread-13.html (you may need to create an account with sr-research)

    Args:
        path_file (str):   Full filepath (with filename) to the edf file to be extracted
        path_output (str): Full filepath (with filename) for the extracted asc file (if .asc extention is missing, it will be added)
    
    Returns:
        path_output (str, the full filepaht to the extracted asc file)
    '''
    if not path_output.endswith('.asc'):
        path_output += '.asc'

    command = 'edf2asc ' + path_file + ' ' + path_output + ' -sg -vel -res -y'
    print('Extracting EDF file (will OVERWRITE any exiting output file). Running:\n\t', command)
    # TODO Include try-catch and report errors
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)
        sys.stdout.flush()
    process.wait()

    return path_output


def asc2hdf( path_file, path_output, get_rowData=False, print_preview=False):
    '''Converts an eyelink ascii file (.asc) to an Heiarchial Data Format (v5) file (.hdf5).

    The HDF output file will have two keys containing data. 
    
        * 'eyelink_annotations' - The messages from the .asc file
        * 'eyelink_samples'     - The numeric samples from the .asc. These are already converted to floats.

    Args:
        path_file (str):                Full filepath (with filename) to the edf file to be extracted
        path_output (str):              Full filepath (with filename) for the extracted asc file (if .asc extention is missing, it will be added)
        get_rowData (bool, optional):   Whether to extract the corresponding sample row index for each annotation message
        print_preview (bool, optional): Whether to print a dataframe preview to stdout during extraction.
    
    Returns:
        path_output (str, the full filepaht to the extracted asc file)
    '''
    print('Reading in ', path_file, '...')
    file_raw          = pd.read_table(path_file, sep='\n', header=None)[0].str.split(expand=True)
    index_annotations = pd.to_numeric(file_raw.iloc[:,0], errors='coerce').isna()
    # print(file_raw)

    print('Extracting Samples...')
    samples  = file_raw.loc[~index_annotations,:]
    samples  = samples.apply( pd.to_numeric, errors='coerce', downcast='float')
    samples.reset_index(inplace=True, drop=True)
    sample_labels = ['timestamp', 'pos_x_left',  'pos_y_left',  'pupil_left',    \
                                  'pos_x_right', 'pos_y_right', 'pupil_right',   \
                                  'vel_x_left',  'vel_y_left',                   \
                                  'vel_x_right', 'vel_y_right',                  \
                                  'res_x',       'res_y',       'input_flags',   \
                                  'head_pos_x',  'head_pos_y',  'head_pos_dist', \
                                  'head_flags'
                     ]
    key_renameColumn = dict( zip( samples.columns, sample_labels))
    samples.rename(key_renameColumn, axis=1, inplace=True)

    print('Extracting Annotations...')
    annotations             = file_raw.loc[index_annotations,:]

    if get_rowData:
        def get_rowData(row):
            print(row.index)
            try:
                return samples[ samples['timestamp'].ge(float(row.iloc[1]))].index[0]
            except:
                for i in range(len(row)):
                    try:
                        return samples['timestamp'].ge(float(row.iloc[i])).index[0]
                    except:
                        continue
            return 0
        annotations['row_data'] = annotations.apply(lambda row: get_rowData(row), axis=1)
    annotations.reset_index(inplace=True, drop=True)

    if print_preview:
        print('Annotations:\n', annotations)
        print('Sample Types:\n', samples.dtypes)
        messages_all = annotations.loc[ annotations.iloc[:,0] ==   'MSG']
        print('\n\nMessages:\n', messages_all.iloc[:,2].unique())
        print('\n\nSamples:  \n', samples)

    if not path_output.endswith('hdf5'):
        path_output += '.hdf5'

    warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
    print('Saving annotations...')
    annotations.to_hdf(path_output, 'eyelink_annotations')
    print('Saving samples...')
    samples.to_hdf(    path_output, 'eyelink_samples')
    print('DONE')

    return path_output


def hdf_getKeys( path_file):
    '''Extracts the existing keys from a .hdf5 file

    Args:
        path_file (str): Full filepath (inluding filename) of the .hdf5 file to analyze.

    Returns:
        keys (list, the existing keys in the dataframe)
    '''
    temp = pd.HDFStore(path_file).keys()
    # Strip off the escape character, not sure why it is there by default
    return [ key[1:] for key in temp ]


def hdf2df( path_file, key_to_get, index_startStop=None):
    '''Imports a .hdf5 file into a pnadas dataframe. Function exists in case any preliminary manipulation is always required before analysis.

    Args:
        path_file (str):  Full filepath (inluding filename) of the .hdf5 file to analyze.
        key_to_get (str): hdf key to extract
        index_startStop (list, optional): Whether to only extract a certain start/stop index. List should contain [index_start, index_stop] indexes.

    Returns:
        dataframe containing data.
    '''
    if index_startStop is None:
        return pd.read_hdf(path_file, key=key_to_get)
    else:
        return pd.read_hdf( path_file, key=key_to_get, start=index_startStop[0], stop=index_startStop[1])


def get_subjectNotes(path_subject, sheetName, filename=None):
    '''Extracts notes about a particular sujbect from an excel file.

    Args:
        path_subject (str):       Path to a subject's raw data folder.
        sheetName (str):          Name of the sheet where the desired notes are kept.
        filename (str, optional): Filename of the data file of interest.  If exists, will look for a 'filename' column in the notes, and only keep rows with a matching filename.

    Returns:
        dataframe containing notes.
    '''
    subject = os.path.basename(os.path.normpath(path_subject.replace('eyetracking','')))
    path_notes = os.path.join( path_subject, subject + '_notes.xlsx')
    
    if os.path.exists(path_notes) and os.path.isfile(path_notes):
        try:
            df_notes = pd.read_excel( path_notes, sheet_name=sheetName, engine='openpyxl')
            if filename is not None:
                df_notes = df_notes.loc[ df_notes['filename'] == filename]
            
            return df_notes
        except ValueError:
            print('ERROR: NOTES: ', subject, '  The sheet ', sheetName, ' may not exist.')
            return None
    
    else:
        print('WARNING: NOTES: No notes file exists ', subject)
        return None


def update_subjectNotes( path_subject, sheetName, df_notes):
    '''Extracts notes about a particular sujbect from an excel file.

    Important:
        This will replace the entire excel sheed with the contents of ``df_notes``. Be careful ALL the desired notes are present.
        Previously exiting notes, if accidentally overwritten/lost, will be irrecoverable.

    Args:
        path_subject (str):          Path to a subject's raw data folder.
        sheetName (str):             Name of the sheet where the desired notes are kept.
        df_notes (pandas dataframe): Full dataframe of notes to replace in the subject notes under the target sheet name.

    Returns:
        Nothing. Notes file will be updated.
    '''
    subject = os.path.basename(os.path.normpath(path_subject.replace('eyetracking','')))
    path_notes = os.path.join( path_subject, subject + '_notes.xlsx')

    df_notes = df_notes.sort_values(by=['filename', 'trial'], ascending=[False, False], inplace=False)

    if os.path.exists(path_notes) and os.path.isfile(path_notes):
        with pd.ExcelWriter( path_notes, mode='a', engine="openpyxl", if_sheet_exists="replace") as writer:
            df_notes.to_excel(writer, sheet_name=sheetName, columns=['filename', 'trial', 'data', 'update', 'timestamp_start', 'timestamp_end', 'value', 'value_new'], index=False)
    else:
        df_notes.to_excel(path_notes, sheet_name=sheetName, columns=['filename', 'trial', 'data', 'update', 'timestamp_start', 'timestamp_end', 'value', 'value_new'], index=False)



def get_wordAlignment_afavaro( subject, session, trial, path_folder, path_rawData=None):
    '''Helper function to import word tokens generated by tcao's token extraction scripts.

    Args:
        subject (str):                Subject of interest
        session (str or int):         Session of interest
        trial (str):                  Trial of interest
        path_folder (str):            Path to location of word alignment output
        path_rawData (str, optional): Path to the location of the raw audio waveforms
    
    Returns
        Dataframe containing information about the word tokens and their timings, as extracted by tcao
    '''
    identifier = '_'.join([subject, session, trial])
    path_alignment = os.path.join( path_folder, identifier) + '.json'
    if not(os.path.exists(path_alignment)) or not(os.path.isfile(path_alignment)):
        print('WARNING: AUDIO: No Alignment for ', identifier)
        return None, None, None
    with open(path_alignment, 'r') as f:
        data = json.load(f)
    data_wordAlign = pd.DataFrame.from_dict(data['word_segments'])
    if data_wordAlign.empty:
        print('WARNING: AUDIO: Empty Alignment for ', identifier)
        return None, None, None
    data_wordAlign['time'] = data_wordAlign['start'].copy()
    # Get the raw data, if exists
    if path_rawData is not None:
        try:
            path_audio     = os.path.join( path_rawData, identifier+'.wav')
            fs, data_audio = wavfile.read(path_audio)
            return data_wordAlign, data_audio, fs
        except:
            print('WARNING: AUDIO: No raw audio found ', identifier)
            return data_wordAlign, None, None
    else:
        return data_wordAlign, None, None



def get_wordAlignment( subject, session, trial, path_folder, path_rawData=None):
    '''Helper function to import word tokens generated by tcao's token extraction scripts.

    Args:
        subject (str):                Subject of interest
        session (str or int):         Session of interest
        trial (str):                  Trial of interest
        path_folder (str):            Path to location of word alignment output
        path_rawData (str, optional): Path to the location of the raw audio waveforms
    
    Returns
        Dataframe containing information about the word tokens and their timings, as extracted by tcao
    '''
    identifier = '_'.join([subject, session, trial])
    path_alignment = os.path.join( path_folder, identifier) + '.csv'
    if not(os.path.exists(path_alignment)) or not(os.path.isfile(path_alignment)):
        print('WARNING: AUDIO: No Alignment for ', identifier)
        return None, None, None
    data_wordAlign = pd.read_csv( path_alignment, index_col=0)
    if data_wordAlign.empty:
        print('WARNING: AUDIO: Empty Alignment for ', identifier)
        return None, None, None
    data_wordAlign  = data_wordAlign.rename( columns={'start_precise': 'time'})
    # Get the raw data, if exists
    if path_rawData is not None:
        try:
            path_audio     = os.path.join( path_rawData, identifier+'.wav')
            fs, data_audio = wavfile.read(path_audio)
            return data_wordAlign, data_audio, fs
        except:
            print('WARNING: AUDIO: No raw audio found ', identifier)
            return data_wordAlign, None, None
    else:
        return data_wordAlign, None, None



def get_wordAlignment_tcao( group, subject, session, trial, path_data_wordTime, path_firstWord=None, path_rawData=None, trial_rawData=None):
    '''Helper function to import word tokens generated by tcao's token extraction scripts.

    Args:
        group (str):                    group of interest
        subject (str):                  Subject of interest
        session (str or int):           Session of interest
        trial (str):                    Trial of interest
        path_data_wordTime (str):       Path to the extracted word token timinng data
        path_firstWord (str, optional): Path to the first word timinng data
        path_rawData (str, optional):   Path to the location of the raw audio waveforms
        trial_rawData (str, optional):  Filename of trial to extract raw data from.
    
    Returns
        Dataframe containing information about the word tokens and their timings, as extracted by tcao
    '''
    # Remove the leading zeros
    subject_tcao = subject.split('_')
    subject_tcao[1] = str(int(subject_tcao[1]))
    subject_tcao = '_'.join(subject_tcao)
    session_tcao = 'ses' + str(int(session[3:]))
    identifier =  '_'.join([subject_tcao, session_tcao, trial])
    # Get the word tokens
    path_wordFirst = os.path.join( path_data_wordTime, identifier+'-word.csv')
    if not(os.path.exists(path_wordFirst)) or not(os.path.isfile(path_wordFirst)):
        print('WARNING: AUDIO: No alignment ', identifier)
        return None, None, None
    data_wordFirst = pd.read_csv( path_wordFirst, index_col=0)
    # Adjust the timing of the first word, if necessary
    if path_firstWord is not None:
        firstWord = pd.read_csv( path_wordFirst, delimiter=' ')
        firstTime = firstWord.loc[ firstWord[:,0] == identifier, 1]
        data_wordFirst.iloc[0].loc['Time'] = firstTime
    # Get the raw data, if exists
    if path_rawData is not None:
        try:
            path_audio     = os.path.join( path_rawData, '_'.join([subject, session, trial_rawData])+'.wav')
            fs, data_audio = wavfile.read(path_audio)
            return data_wordFirst, data_audio, fs
        except:
            print('WARNING: AUDIO: No raw audio found ', identifier)
            return data_wordFirst, None, None
    else:
        return data_wordFirst, None, None



def get_processedEye(path_to_parent_folder, hdf_key, hdf_key_note='', subject_include=None):
    '''Helper function to extract all of the processed data for all subjects in the Neuro-Logical patient cohort.
    Assumes a particular structure of proccessed output exists according to the way Trevor Meyer saves eyetracking data.

    Args:
        path_to_parent_folder (str):      Path to a subject's raw data folder.
        hdf_key (str):                    Name of the sheet where the desired notes are kept.
        hdf_key_note (str, optional):     Full dataframe of notes to replace in the subject notes under the target sheet name.
        subject_include (list, optional): List of subjects to include.

    Returns:
        summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLsub
    '''

    if (len(hdf_key_note) > 0) and not( hdf_key_note.startswith('_')):
        hdf_key_note = '_' + hdf_key_note
        
    path_allData = os.path.join( path_to_parent_folder, 'data_summary.hdf')
    if os.path.exists( path_allData):
        summary_ALLsac = pd.read_hdf( path_allData, hdf_key+'/saccade' +hdf_key_note)
        summary_ALLfix = pd.read_hdf( path_allData, hdf_key+'/fixation'+hdf_key_note)
        summary_ALLblk = pd.read_hdf( path_allData, hdf_key+'/blink'   +hdf_key_note)
        summary_ALLsub = pd.read_hdf( path_allData, hdf_key+'/summary' +hdf_key_note)
        try:
            summary_ALLgaz = pd.read_hdf( path_allData, hdf_key+'/gaze'    +hdf_key_note)
        except:
            print('WARNING: GAZE: No gaze data found for ', subject)
            summary_ALLgaz = None
        try:
            summary_ALLwrd = pd.read_hdf( path_allData, hdf_key+'/wordBegin'+hdf_key_note)
        except:
            print('WARNING: AUDIO: No wordTime data found for ', subject)
            summary_ALLwrd = None

        return summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd, summary_ALLsub


    # Initialize final output dataframes
    summary_ALLsac = pd.DataFrame()
    summary_ALLfix = pd.DataFrame()
    summary_ALLblk = pd.DataFrame()
    summary_ALLgaz = pd.DataFrame()
    summary_ALLwrd = pd.DataFrame()
    summary_ALLsub = pd.DataFrame()

    # Iterate through each experimental group
    for group in sorted( os.listdir(path_to_parent_folder), reverse=True):
        path_group = os.path.join( path_to_parent_folder, group)
        if os.path.isdir(path_group):

            # Iterate through each subject in the experimental group
            for subject in os.listdir(path_group):
                if (subject_include is not None) and (subject not in subject_include):
                    continue
                path_subject = os.path.join( path_group, subject)
                if os.path.isdir(path_subject):
                    path_subject = os.path.join( path_group, subject)

                    # Initialize output dataframes for this subject
                    summary_saccade  = pd.DataFrame()
                    summary_fixation = pd.DataFrame()
                    summary_blink    = pd.DataFrame()
                    summary_gaze     = pd.DataFrame()
                    summary_word     = pd.DataFrame()
                    summary_subject  = pd.DataFrame()

                    # Check if output was alrady extracted, or if it must be (re)processed
                    path_processed = os.path.join( path_subject, subject+'_info.hdf')
                    hdfstore_dir_exists = False
                    if os.path.isfile(path_processed):
                        store = pd.HDFStore(path_processed)
                        hdfstore_dir_exists = any( [s.startswith('/'+hdf_key+'/') for s in store.keys()])
                        store.close()
                    if hdfstore_dir_exists:
                        print('Loading ', subject, '...')
                        summary_saccade  = pd.read_hdf( path_processed, hdf_key+'/saccade' +hdf_key_note)
                        summary_fixation = pd.read_hdf( path_processed, hdf_key+'/fixation'+hdf_key_note)
                        summary_blink    = pd.read_hdf( path_processed, hdf_key+'/blink'   +hdf_key_note)
                        summary_subject  = pd.read_hdf( path_processed, hdf_key+'/summary' +hdf_key_note)
                        try:
                            summary_gaze = pd.read_hdf( path_processed, hdf_key+'/gaze'    +hdf_key_note)
                        except:
                            print('WARNING: GAZE: No gaze data found for ', subject)
                            summary_gaze = pd.DataFrame()
                        try:
                            summary_word = pd.read_hdf( path_processed, hdf_key+'/wordBegin'+hdf_key_note)
                        except:
                            print('WARNING: AUDIO: No wordTime data found for ', subject)
                            summary_word = pd.DataFrame()

                        if 'group' in summary_saccade.columns:
                            summary_saccade.loc[ :,'group'] = group
                            summary_fixation.loc[:,'group'] = group
                            summary_blink.loc[   :,'group'] = group
                            summary_subject.loc[ :,'group'] = group
                            if len(summary_gaze) > 0:
                                summary_gaze.loc[    :,'group'] = group
                            if len(summary_word) > 0:
                                summary_word.loc[    :,'group'] = group

                    # Append this subjects data to the overall summary array
                    summary_ALLsac = summary_ALLsac.append(summary_saccade)
                    summary_ALLfix = summary_ALLfix.append(summary_fixation)
                    summary_ALLblk = summary_ALLblk.append(summary_blink)
                    summary_ALLgaz = summary_ALLgaz.append(summary_gaze)
                    summary_ALLwrd = summary_ALLwrd.append(summary_word)
                    summary_ALLsub = summary_ALLsub.append(summary_subject)

    return summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd, summary_ALLsub


def get_processedAudio( path_to_parent_folder):
    '''Helper function to extract all of the processed data for all subjects in the Neuro-Logical patient cohort.
    Assumes a particular structure of proccessed output exists according to the way Anna Favaro saves speech processing output.

    Args:
        path_to_parent_folder (str):      Path to a subject's raw data folder.

    Returns:
        df_intensity, df_pause, df_prosody, df_rhythm, df_ALLsub
    '''
    df_intensity = pd.read_csv( os.path.join( path_to_parent_folder, 'intensity.csv'), index_col=0)
    df_pause     = pd.read_csv( os.path.join( path_to_parent_folder, 'pause.csv'), index_col=0)
    df_prosody   = pd.read_csv( os.path.join( path_to_parent_folder, 'prosody.csv'), index_col=0)
    df_rhythm    = pd.read_csv( os.path.join( path_to_parent_folder, 'rhythm.csv'), index_col=0)

    df_intensity['filename'] = df_intensity.apply( lambda r: os.path.splitext( os.path.basename(r['sound_filepath']))[0], axis=1)
    df_pause['filename']     = df_pause.apply(     lambda r: os.path.splitext( os.path.basename(r['AudioFile']))[0], axis=1)
    df_prosody['filename']   = df_prosody.apply(   lambda r: os.path.splitext( os.path.basename(r['id']))[0], axis=1)
    df_rhythm['filename']    = df_rhythm.apply(    lambda r: os.path.splitext( os.path.basename(r['name']))[0], axis=1)

    df_ALLsub = df_intensity.merge( df_pause, how='outer', on='filename').merge(df_prosody, how='outer', on='filename').merge(df_rhythm, how='outer', on='filename')
    
    return df_intensity, df_pause, df_prosody, df_rhythm, df_ALLsub



def get_data_raw(path_to_parent_folder, hdf_key, hdf_key_note='', filename_target=None, reextract=False):
    '''Helper function to extract summary data for all subjects in the Neuro-Logical patient cohort.
    Assumes a particular structure of proccessed output exists according to the way Trevor Meyer saves eyetracking data.

    Args:
        path_to_parent_folder (str):     Path to a subject's raw data folder.
        hdf_key (str):                   Name of the sheet where the desired notes are kept.
        hdf_key_note (str, optional):    Full dataframe of notes to replace in the subject notes under the target sheet name.
        filename_target (str, optional): Specific filename to target.  All filenames that do not match exactly will be skipped.
        reextract (bool, optional):      Whether to reextract the data. Otherwise will trust any existing extractions and only reextract if needed.

    Returns:
        summary_ALLsub
    '''

    if (len(hdf_key_note) > 0) and not( hdf_key_note.startswith('_')):
        hdf_key_note = '_' + hdf_key_note
        
    # Initialize final output dataframes
    summary_ALLsub = pd.DataFrame()

    # Iterate through each experimental group
    for group in sorted( os.listdir(path_to_parent_folder), reverse=True):
        path_group = os.path.join( path_to_parent_folder, group)
        if os.path.isdir(path_group):

            # Iterate through each subject in the experimental group
            for subject in os.listdir(path_group):
                path_subject = os.path.join( path_group, subject)
                if os.path.isdir(path_subject):
                    path_subject = os.path.join( path_group, subject)
                    
                    for index_file, filename in enumerate( os.listdir(path_subject)):
                        if (filename_target is not None) and (filename != filename_target):
                            continue

                        if filename.lower().endswith('.edf'):
                            print('Subject:\t', '\t\t'.join([subject,group,filename]), ' - ', index_file)

                            # Extract the raw data from the edf file, if necessary
                            path_raw     = os.path.join( path_subject, filename)
                            path_extract = os.path.splitext( path_raw)[0] + '.hdf5'

                            hdfstore_dir_exists = False
                            if os.path.isfile(path_extract):
                                store = pd.HDFStore(path_extract)
                                hdfstore_dir_exists = any( [s.startswith('/'+hdf_key+'/') for s in store.keys()])
                                store.close()
                            if not( hdfstore_dir_exists) or reextract:
                                path_intermediate = os.path.splitext( path_raw)[0] + '.asc'
                                if not( os.path.exists(path_intermediate)) or reextract:
                                    edf2asc(path_raw, path_intermediate)
                                asc2hdf(path_intermediate, path_extract)

                            data_eye_annotation = hdf2df( path_extract, hdf_key)
                            
                            if 'filename' not in data_eye_annotation.columns:
                                data_eye_annotation.loc[:,'filename'] = filename
                            if 'subject' not in data_eye_annotation.columns:
                                data_eye_annotation.loc[:,'subject']  = subject
                            if 'group' not in data_eye_annotation.columns:
                                data_eye_annotation.loc[:,'group']    = group

                            # Append this subjects data to the overall summary array
                            summary_ALLsub.append( data_eye_annotation)

    return summary_ALLsub


if __name__ == '__main__':

    source_edf = './NLS_6.edf'
    inter_asc  = os.path.splitext(source_edf)[0] + '.asc'
    final_hdf  = os.path.splitext(source_edf)[0] + '.hdf5'


    edf2asc(source_edf, inter_asc)
    asc2hdf( inter_asc, final_hdf)
    data = hdf2df(final_hdf,'eyelink_samples')

    print('FINAL_DATA:\n', data)
    print(data.dtypes)
