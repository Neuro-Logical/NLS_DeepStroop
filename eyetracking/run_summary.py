from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

try:
    import eyetracking.functions.extract   as extract
    import eyetracking.functions.summarize as summarize
except:
    import functions.extract   as extract
    import functions.summarize as summarize

sns.set(font_scale=2.2)
sns.set(font_scale=1.2)
sns.set_style('whitegrid')



def main():
    # Define File Paths
    # Eyetracking Analysis (From Trevor)
    # path_processedEye         = os.path.join(path_output, 'data_processed')
    path_processedEye         = '/Users/trevor/sync/clsp_b15/eyelink_preliminaryRun04'
    hdf_key_dir               = 'tmeyer'
    # Audio Analysis (From Anna)
    path_processedAudio   = '/Users/trevor/datasets/eyelink/audio/afavaro_processed'
    # Subject Metadata (From RedCap)
    path_metadata          = '/Users/trevor/gitrepos/eyetracking/metadata/0.metadata.csv'
    # Metric Output
    path_titles            = '/Users/trevor/gitrepos/eyetracking/publication/metricTitles.csv'

    # Final Output
    # path_output            = '/Users/trevor/Dropbox/Mac (2)/Documents/outputs/eyelink'
    # path_output            = '/home/trevor-debian/Documents/outputs/eyetracking_output'
    # path_output            = '/export/b15/tmeyer16/outputs/eyelink'
    path_output            = '/Users/trevor/sync/clsp_b15/eyelink_preliminaryRun04'

    subfolder_output       = 'test01'
    folder_output_stats    = 'stats'
    folder_output_corr     = 'correlation'
    folder_output_boxplots = 'boxplots'
    folder_output_profile  = 'profile'

    # Define Trials and Groups
    use_trialContaining  = '-naming'
    use_groupsContaining = ['CTL', 'AD', 'PD']
    use_staticGroups     = True
    use_groupsForPub     = True
    # Create Subgroups
    subgroups            = ''
    subgroup_thresholds  = None
    # subgroups            = 'H&Y'
    # subgroup_thresholds  = ['2.5','99']
    # subgroup_thresholds  = ['2','99']
    # subgroups            = 'UPDRS3'
    # subgroup_thresholds  = ['25', '99']
    # subgroups            = 'certainty'
    # subgroup_thresholds  = None
    # subgroups              = 'moca'
    # subgroup_thresholds    = ['20', '99']
    
    # Operations to perfrom
    recalculate_metricSummary = False
    include_trialDifferences  = True
    include_stroopDifferences = True
    calc_statistics           = True
    plot_correlations         = True
    plot_boxplots             = True
    plot_featureProfile       = False

    # Output Definitions
    show_plots                = False
    save_outputs              = True
    age_boundary              = 5
    min_analyzeGroupSize      = 10
    pvalue_threshold          = 0.05

    boxplot_order_group     = [ 'PDM', 'PD', 'CTL', 'AD']
    boxplot_order_trial     = ['color-naming', 'word-naming', 'word-color-naming']
    boxplot_order_trialDiff = ['word-naming-MINUS-color-naming', 'word-color-naming-MINUS-color-naming', 'word-color-naming-MINUS-word-naming']

    show_boxplotWith     = [' ', 'mean', 'std', 'med', 'count', 'time', 'focus', 'rate', 'repeat', 'numBlinks_beforeFocus']
    # show_corrPlotsWith   = ['age', 'moca', 'cdr', 'UPDRS3', 'h&y']
    show_corrPlotsWith   = ['moca', 'UPDRS3']

    # profile_metrics      = ['Avg Number Visits', 'Trial Length', 'Word Rate', 'Number of Fixations', 'Blink Rate']
    profile_metrics      = ['Number Fixations per Word Variability','Duration Fixating per Word Variability','Number Saccades','Number of Fixations','Avg Number Fixations per Word','Time Look before Speaking Variance','Speech Pause Percent','Med Gaze Position Variability','Duration Focusing','Rhythm Variability','Avg Time Looking at Each Word','Avg Gaze Position Variability','Avg Duration Fixating per Word','F0 Variability','Avg Time Look before Speaking','Duration Blinking per Word Variability','Number Visits Variability','Avg Duration Blinking per Word','Avg Number Visits','Avg Number Blinks per Word','Avg Time Look after Speaking','Proportion of Trial Blinking','Blink Rate','Avg Blink Duration','Word Rate','Percentage Words in Vocabulary','Repeated Words per Time','Avg Exiting VelocityMax','Avg Entering VelocityMax','Gaze Words per Trial Length','Avg Entering VelocityAvg','Avg Fixation Duration per Word','Intensity Variability','Avg Saccade VelocityAvg','Percent Words Correct','Avg Saccade Distance','Saccade Distance Variability']


    '''
        Define output paths
    '''
    filetree_stats           = [ folder_output_stats,    subfolder_output]
    filetree_corrSummary     = [ folder_output_corr,     subfolder_output, 'summary']
    filetree_corrDifferences = [ folder_output_corr,     subfolder_output, 'trialChanges']
    filetree_corrStroop      = [ folder_output_corr,     subfolder_output, 'stroopMetrics']
    filetree_boxSummary      = [ folder_output_boxplots, subfolder_output, 'summary']
    filetree_boxDifferences  = [ folder_output_boxplots, subfolder_output, 'trialChanges']
    filetree_boxStroop       = [ folder_output_boxplots, subfolder_output, 'stroopMetrics']
    filetree_profSummary     = [ folder_output_profile,  subfolder_output, 'summary']
    filetree_profDifferences = [ folder_output_profile,  subfolder_output, 'trialChanges']
    filetree_profStroop      = [ folder_output_profile,  subfolder_output, 'stroopMetrics']
    filetree_boxRaw          = [ folder_output_boxplots, subfolder_output, 'raw']

    for filetree in [filetree_stats, filetree_corrSummary, filetree_corrDifferences, filetree_corrStroop, filetree_boxSummary, filetree_boxDifferences, filetree_boxStroop, filetree_profSummary, filetree_profDifferences, filetree_profStroop, filetree_boxRaw]:
        target = path_output
        for newTarget in filetree:
            target = os.path.join( target, newTarget)
            if not( os.path.exists(target)) or not( os.path.isdir(target)):
                os.mkdir(target)

    path_out_stats           = os.path.join( path_output, *filetree_stats)
    path_out_corrSummary     = os.path.join( path_output, *filetree_corrSummary)
    path_out_corrDifferences = os.path.join( path_output, *filetree_corrDifferences)
    path_out_corrStroop      = os.path.join( path_output, *filetree_corrStroop)
    path_out_boxSummary      = os.path.join( path_output, *filetree_boxSummary)
    path_out_boxDifferences  = os.path.join( path_output, *filetree_boxDifferences)
    path_out_boxStroop       = os.path.join( path_output, *filetree_boxStroop)
    path_out_profSummary     = os.path.join( path_output, *filetree_profSummary)
    path_out_profDifferences = os.path.join( path_output, *filetree_profDifferences)
    path_out_profStroop      = os.path.join( path_output, *filetree_profStroop)
    path_out_boxRaw          = os.path.join( path_output, *filetree_boxRaw)


    #############################################
    ##         BEGIN DATA IMPORT STEPS         ##
    #############################################
    '''
        Import and Combine Raw Data
    '''
    # Get eyetracking output
    summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd, summary_ALLeye = extract.get_processedEye(path_processedEye, hdf_key_dir)
    # Get speech output
    summary_ALLint, summary_ALLpau, summary_ALLpro, summary_ALLrhy, summary_ALLspe = extract.get_processedAudio( path_processedAudio)
    summary_ALLspe = summary_ALLspe.add_prefix('speech_')
    # Merge eyetracking and speech
    summary_ALLeye['trialForMerge'] = summary_ALLeye.apply( lambda row: '_'.join(( row['subject'],  row['trial_index'][:-2], row['trial'])), axis=1)
    summary_ALLspe['trialForMerge'] = summary_ALLspe.apply( lambda row: row['speech_filename'][:-4].replace('WordColor', 'word-color-naming').replace('SecuencestroopPrevious1', 'word-naming').replace('SecuencestroopPrevious2', 'color-naming'), axis=1)
    summary_ALL = summary_ALLeye.merge( summary_ALLspe, on='trialForMerge', how='outer').drop_duplicates('trialForMerge', keep='last').reindex()
    summary_ALL['uniqueID']  = summary_ALL['subject'].astype(str) + '+' + summary_ALL['trial_index'].astype(str)
    summary_ALL['sessionID'] = summary_ALL['subject'].astype(str) + '_' + summary_ALL['session'].astype(str)
    # duplicated_trials = summary_ALL[ summary_ALL['trialForMerge'].duplicated()]['trialForMerge']
    df_metadata = pd.read_csv(path_metadata).set_index('subject')
    df_metadata['subject'] = df_metadata.index

    # Add trail labels for speech data that had no eyetracking output
    summary_ALL['group']   = 'UKN' # Reset all groups to use labels
    # print( sum(summary_ALL['date_recorded'].isna()), ' audio trials are missing eyetracking output.') # TODO Not sure what this does
    # for i, row in summary_ALL[ summary_ALL['date_recorded'].isna()].iterrows():
    for i, row in summary_ALL.iterrows():
        trial_parts = row['trialForMerge'].split('_')
        sub = trial_parts[0] + '_' + trial_parts[1]
        if sub in df_metadata.index:
            summary_ALL.loc[i,'group']         = df_metadata.loc[sub, 'label']
        summary_ALL.loc[i,'subject']       = sub
        summary_ALL.loc[i,'trial']         = trial_parts[3]
        summary_ALL.loc[i,'trial_index']   = trial_parts[2]

    # Merge Metadata
    for mdata in df_metadata.columns:
        if mdata == 'subject':
            continue
        if mdata.lower() in ['moca','updrs3']:
            delimiter = ';'
            summary_ALL[mdata] = np.nan
            for i, row in summary_ALL.iterrows():
                if row['subject'] not in df_metadata.index:
                    continue
                if delimiter in str(df_metadata.loc[ row['subject'], mdata]):
                    index = int(row['session'][3:]) -1
                    scores = str(df_metadata.loc[ row['subject'], mdata]).strip().split(';')
                    if (index < len(scores)) and scores[index].isnumeric():
                        summary_ALL.loc[i,mdata] = scores[index]
                else:
                    summary_ALL.loc[i,mdata] = df_metadata.loc[ row['subject'], mdata]
        else:
            summary_ALL[mdata] = summary_ALL.apply( lambda row: df_metadata.loc[ row['subject'], mdata] if row['subject'] in df_metadata.index else '-1', axis=1)
        
        try:
            summary_ALL[mdata] = summary_ALL[mdata].astype(float)
        except:
            print('WARNING: Unable to convert ', mdata, ' to float type.')

    '''
        Group and stratify subjects and trials for comparison
    '''
    # Create subgroups, if requested
    if len(subgroups) > 0:
        folder_output_stats    += '_' + subgroups
        folder_output_boxplots += '_' + subgroups
        summary_ALL = summarize.make_subgroups(summary_ALL, df_metadata, subgroup=subgroups, thresholds=subgroup_thresholds)
    
    # Define all groups to analyze
    groups_ALL = summary_ALL['group'].dropna().unique().tolist()
    if use_groupsContaining is not None:
        groups_ALL = [grp for grp in groups_ALL if any([g in grp for g in use_groupsContaining])]

    # Decide which groups/subjects to keep
    if not use_staticGroups:
        summary_ALL = summary_ALL[  summary_ALL['procSuccess_mvmt'] \
                                  & summary_ALL['procSuccess_gaze'] \
                                  & summary_ALL['procSuccess_word'] \
                                  & summary_ALL['age'].notna()      \
                                ]
    if use_groupsForPub:
        # group_AIM = {}
        # group_AIM['CTL']  = [ 'AD_002','NLS_006','NLS_026','NLS_073','NLS_075','PEC_001','PEC_003','PEC_006','PEC_007','PEC_010','PEC_012','PEC_013','PEC_020','PEC_028','PEC_030','PEC_031','PEC_032','PEC_034','PEC_037','PEC_038','PEC_039','PEC_040','PEC_042','PEC_043']
        # group_AIM['CTLb'] = ['NLS_019','NLS_023','NLS_024','NLS_026','NLS_075','NLS_077','PEC_006','PEC_012','PEC_013','PEC_020','PEC_028','PEC_031','PEC_034','PEC_037'] # For use when comparing PDM group
        # group_AIM['PD']   = [ 'AD_005','NLS_005','NLS_007','NLS_010','NLS_012','NLS_016','NLS_017','NLS_022','NLS_033','NLS_043','NLS_046','NLS_056','NLS_063','NLS_068','NLS_069','NLS_074','NLS_076','NLS_081','NLS_085','NLS_087','NLS_097','NLS_098','NLS_102']
        # group_AIM['PDM']  = ['NLS_034','NLS_041','NLS_042','NLS_044','NLS_047','NLS_055','NLS_058','NLS_080','NLS_088','NLS_090']
        # group_AIM['AD']   = [ 'AD_001', 'AD_003', 'AD_004', 'AD_006', 'AD_007', 'AD_008', 'AD_009', 'AD_010', 'AD_011',' AD_012', 'AD_014']
        # group_AIM['CTL'] = ['NLS_077','NLS_006','PEC_034','PEC_037','PEC_031','NLS_075','PEC_027','PEC_003','PEC_007','PEC_042','PEC_001','PEC_038','AD_002','AD_002','PEC_039','PEC_040','PEC_012','PEC_006','PEC_013','PEC_032','PEC_011','PEC_028','PEC_021','PEC_045','PEC_062','NLS_073','PEC_043','PEC_020','PEC_056','PEC_002']
        # group_AIM['CTLb'] = ['NLS_024','NLS_019','NLS_023','NLS_077','NLS_006','PEC_034','PEC_037','PEC_031','NLS_075','PEC_042','PEC_038','AD_002','PEC_039','PEC_012','PEC_006','PEC_013','PEC_028','PEC_021','PEC_045','PEC_062','NLS_073','PEC_043']
        # group_AIM['PD'] = ['NLS_037','NLS_035','NLS_015','NLS_004','NLS_055','NLS_010','NLS_081','NLS_097','NLS_056','NLS_034','NLS_085','NLS_098','NLS_087','NLS_049','NLS_022','NLS_046','NLS_101','NLS_050','NLS_069','NLS_076','NLS_102','NLS_043','NLS_063','NLS_074','NLS_016','NLS_068','NLS_005','NLS_017','NLS_012','NLS_033','NLS_103','NLS_113','NLS_120','NLS_124']
        # group_AIM['PDM'] = ['NLS_090','NLS_060','NLS_053','NLS_053','NLS_058','NLS_047','NLS_041','NLS_007','NLS_007','NLS_042','NLS_080','NLS_080','AD_005','AD_005','AD_005','NLS_119']
        # group_AIM['AD'] = ['AD_006','AD_019','AD_019','AD_003','AD_003','AD_003','AD_011','AD_011','AD_010','AD_004','AD_004','AD_004','AD_004','AD_001','AD_001','AD_008','AD_008','AD_009','AD_009','AD_021','AD_014','AD_014','AD_007','AD_007','AD_007','AD_015','AD_016','AD_018','AD_022','AD_023','AD_024']

        group_pub = {}

        # Good
        group_pub['AD']   = ['AD_003_ses02','AD_003_ses03','AD_004_ses01','AD_004_ses02','AD_004_ses03','AD_007_ses03','AD_008_ses03','AD_009_ses02','AD_009_ses03','AD_011_ses01','AD_012_ses01','AD_012_ses02','AD_014_ses01','AD_014_ses02','AD_014_ses03','AD_016_ses01','AD_018_ses02','AD_023_ses01','AD_024_ses01']
        group_pub['PD']   = ['NLS_004_ses01','NLS_004_ses02','NLS_005_ses01','NLS_010_ses01','NLS_015_ses01','NLS_016_ses01','NLS_017_ses01','NLS_022_ses01','NLS_034_ses01','NLS_035_ses01','NLS_046_ses01','NLS_049_ses01','NLS_050_ses01','NLS_055_ses01','NLS_056_ses01','NLS_063_ses01','NLS_068_ses01','NLS_069_ses01','NLS_076_ses01','NLS_085_ses01','NLS_095_ses01','NLS_097_ses01','NLS_098_ses01','NLS_102_ses01','NLS_103_ses01','NLS_104_ses01','NLS_127_ses01','NLS_130_ses01','NLS_138_ses01','NLS_142_ses01','NLS_169_ses01']
        group_pub['PDM']  = ['AD_005_ses01','AD_005_ses02','NLS_032_ses01','NLS_041_ses01','NLS_042_ses01','NLS_044_ses01','NLS_045_ses01','NLS_057_ses01','NLS_058_ses01','NLS_060_ses01','NLS_065_ses01','NLS_079_ses01','NLS_090_ses01','NLS_094_ses01','NLS_119_ses01','NLS_135_ses01']
        group_pub['CTL']  = ['AD_017_ses01','AD_017_ses02','AD_025_ses01','NLS_006_ses01','NLS_018_ses01','NLS_023_ses01','NLS_073_ses01','NLS_075_ses01','NLS_105_ses01','NLS_106_ses01','NLS_107_ses01','NLS_108_ses01','NLS_111_ses01','NLS_112_ses01','PEC_007_ses01','PEC_011_ses01','PEC_013_ses01','PEC_022_ses01','PEC_027_ses01','PEC_039_ses01','PEC_042_ses01','PEC_043_ses01','PEC_046_ses01','PEC_049_ses01','PEC_050_ses01','PEC_057_ses01','PEC_059_ses01','PEC_065_ses01']
        group_pub['CTLa'] = ['NLS_106_ses01','PEC_007_ses01','PEC_011_ses01','PEC_027_ses01','PEC_043_ses01','PEC_046_ses01','PEC_049_ses01','PEC_057_ses01','PEC_059_ses01','PEC_065_ses01','NLS_006_ses01','NLS_073_ses01','NLS_075_ses01','NLS_111_ses01','PEC_013_ses01','PEC_022_ses01','PEC_039_ses01','PEC_042_ses01','PEC_050_ses01']
        group_pub['CTLb'] = ['AD_017_ses01','AD_017_ses02','AD_025_ses01','NLS_006_ses01','NLS_073_ses01','NLS_075_ses01','NLS_105_ses01','NLS_106_ses01','NLS_107_ses01','NLS_108_ses01','NLS_111_ses01','PEC_007_ses01','PEC_011_ses01','PEC_013_ses01','PEC_022_ses01','PEC_027_ses01','PEC_039_ses01','PEC_042_ses01','PEC_043_ses01','PEC_046_ses01','PEC_049_ses01','PEC_050_ses01','PEC_057_ses01','PEC_059_ses01','PEC_065_ses01']
        group_pub['CTLc'] = ['AD_017_ses01','AD_017_ses02','AD_025_ses01','NLS_018_ses01','NLS_023_ses01','NLS_105_ses01','NLS_106_ses01','NLS_107_ses01','NLS_108_ses01','PEC_007_ses01','PEC_011_ses01','PEC_027_ses01','PEC_043_ses01','PEC_046_ses01','PEC_049_ses01','PEC_057_ses01','PEC_059_ses01','PEC_065_ses01']
        # Okay
        group_pub['AD'].extend(  ['AD_001_ses02','AD_007_ses01','AD_007_ses02','AD_009_ses01','AD_013_ses01','AD_013_ses03','AD_015_ses01','AD_018_ses01','AD_021_ses01','AD_022_ses01'])
        group_pub['PD'].extend(  ['NLS_012_ses01','NLS_033_ses01','NLS_074_ses01','NLS_113_ses01','NLS_124_ses01','NLS_162_ses01','NLS_171_ses01'])
        group_pub['PDM'].extend( ['AD_005_ses03','NLS_047_ses01','NLS_080_ses01','NLS_091_ses01','NLS_125_ses01','NLS_134_ses01'])
        group_pub['CTL'].extend( ['AD_002_ses03','NLS_024_ses01','NLS_026_ses01','NLS_123_ses01','PEC_010_ses01','PEC_034_ses01','PEC_040_ses01','PEC_045_ses01'])
        group_pub['CTLa'].extend(['AD_002_ses03','PEC_010_ses01','PEC_034_ses01','PEC_040_ses01','PEC_045_ses01'])
        group_pub['CTLb'].extend(['AD_002_ses03','NLS_026_ses01','PEC_010_ses01','PEC_034_ses01','PEC_040_ses01','PEC_045_ses01'])
        group_pub['CTLc'].extend(['NLS_024_ses01','NLS_026_ses01','NLS_123_ses01','PEC_034_ses01','PEC_045_ses01'])

        summary_ALL['group_pub'] = False
        for key, value in group_pub.items():
            summary_ALL.loc[ summary_ALL['sessionID'].isin(value), 'group_pub'] = True
        summary_ALL = summary_ALL.loc[ summary_ALL['group_pub'], :]

    # Define All trials to analyze
    trials_ALL = [t for t in summary_ALL['trial'].unique() if use_trialContaining in t]

    '''
        Apply acceptance criteria
    '''
    # Controls must have MoCA > 25
    summary_ALL = summary_ALL.loc[ ~((summary_ALL['group'] == 'CTL') & (summary_ALL['moca'] < 25)), : ]
    # Exclude UPDRS3 scores > 60
    summary_ALL = summary_ALL[ (summary_ALL['UPDRS3'] < 50) | (summary_ALL['UPDRS3'].isna())]

    '''
        Define metrics to analyze
    '''
    # Convert Titles to output format
    if os.path.isfile(path_titles):
        titles = pd.read_csv(path_titles, header=0)
        title_all  = titles.loc[ titles['title'] != 'ignore', 'title'].to_list()
        metric_all = titles.loc[ titles['title'] != 'ignore', 'metric'].to_list()
        title_mapper    = dict( zip( titles['metric'], titles['title']))
        title_mapperRev = dict( zip( titles['title'], titles['metric']))
        category_mapper = dict( zip( titles['title'], titles['category']))
    
    def in_metricsToSkip(metric):
        # return not( metric in ['Duration Blinking per Word normalized'])
        # return not( metric.startswith('speech_'))
        # return not( 'Time Until' in metric)
        if any( [s in metric for s in ['timestamp']]):
            return True
        return not( metric in metric_all)
        # return  ('left' in metric) or ('right' in metric) or ('perc-' in metric) or ('timestamp' in metric) or \
        #         ('gaze_line' in metric) or ('gaze_word' in metric) or ('gaze_index' in metric) or \
        #         (('gaz' in metric) and ('perTimeTrial' in metric)) or \
        #         ('subject' in metric) or ('group' in metric) or ('trial' in metric) or ('test' in metric) or metric.strtswith('procSuccess_')\
        #         (('pos' in metric) and (('x' in metric) or ('y' in metric))) or \
        #         ('line-' in metric.lower()) or ('word-' in metric.lower())
    

    # Save metadata to output folder, for reference
    summary_ALL.loc[                            :,['subject','group','trial','trial_index','age','sex','moca','UPDRS3','cdr']].to_csv( os.path.join( path_output, 'metadata_subject.csv'))
    summary_ALL.drop_duplicates('uniqueID').loc[:,['subject','group','trial','trial_index','age','sex','moca','UPDRS3','cdr']].to_csv( os.path.join( path_output, 'metadata_trial.csv'))


    #############################################
    ##       BEGIN SINGLE TRIAL ANALYSIS       ##
    #############################################
    output_stat = []
    output_corr = []
    # Analyze all the metrics separately
    for metric in sorted( summary_ALL.columns):
        if in_metricsToSkip(metric) or (metric in df_metadata.columns) or not(pd.api.types.is_numeric_dtype( summary_ALL[metric])):
            continue
        # array for logging text
        output_text  = []

        # Log output progress
        if metric in title_mapper.keys():
            output_text.append(title_mapper[metric] + ' (' + metric + ')')
            print('Analyzing ', title_mapper[metric] + ' (' + metric + ')')
        else:
            output_text.append(metric)
            print('Analyzing ', metric)

        # Plot Boxplots of Metric
        if plot_boxplots and any([ m in metric for m in show_boxplotWith]):
            summary_ALLsub_ALLtrials = summary_ALL[ summary_ALL['trial'].isin(trials_ALL)].copy()
            summarize.get_boxplots(summary_ALLsub_ALLtrials, metric, group_order=boxplot_order_group, trial_order=boxplot_order_trial, path_output=path_out_boxSummary, save_outputs=save_outputs, show_plots=show_plots)

        # Iterate through all trials
        for trial in trials_ALL:
            # Log output progress
            output_text.append('\t' + str(trial))
            summary_header = False
            print('\t', trial)

            # Separate out the trial of interest
            summary_ALL_indexTrial = (summary_ALL['trial'] == trial)

            # Compare all the groups (group "i" and group "j") to one another
            for i in range(len(groups_ALL)):
                # Define group "i"
                group_i    = summary_ALL.loc[ summary_ALL_indexTrial & (summary_ALL['group'] == groups_ALL[i]), :]
                ages_use_i = summarize.get_ageInclude( group_i, age_boundary)
                
                # Plot correlations
                if plot_correlations and (len(group_i) >= min_analyzeGroupSize):
                    result_corr = summarize.get_correlations( group_i, metric, show_corrPlotsWith, trial + ' ' + groups_ALL[i], title_mapper=title_mapper, normalize=False, pvalue_threshold=pvalue_threshold, path_output=path_out_corrSummary, save_outputs=save_outputs, save_prefix=groups_ALL[i], show_plots=show_plots)
                    for _, row in result_corr.iterrows():
                        temp = row.to_dict()
                        temp.update({'trial': trial, 'cnt': len(group_i), 'group': groups_ALL[i]})
                        output_corr.append(temp)

                if calc_statistics and (len(group_i) >= min_analyzeGroupSize):
                    for j in range(i+1, len(groups_ALL)):
                        # define group "j"
                        group_j = summary_ALL.loc[ (summary_ALL_indexTrial & (summary_ALL['group'] == groups_ALL[j])), [metric,'subject','sessionID','age']]
                        ages_use = summarize.get_ageInclude( group_j, age_boundary, existing_ages=ages_use_i)

                        # Filter groups for age/publication/matched group constraings
                        if not use_staticGroups:
                            group_i_forJ = group_i.loc[ group_i['age'].isin(ages_use),metric].dropna().to_numpy()
                            group_j      = group_j.loc[ group_j['age'].isin(ages_use),metric].dropna().to_numpy()
                        else:
                            if use_groupsForPub:
                                if (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'AD'):
                                    group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PD'):
                                    group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PDM'):
                                    group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                else:
                                    group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub[groups_ALL[i]]),metric].dropna().to_numpy()
                                if (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'AD'):
                                    group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PD'):
                                    group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PDM'):
                                    group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                else:
                                    group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub[groups_ALL[j]]),metric].dropna().to_numpy()
                            else:
                                group_i_forJ = group_i.loc[ :,metric].dropna().to_numpy()
                                group_j      = group_j.loc[ :,metric].dropna().to_numpy()

                        if (len(group_i_forJ) >= min_analyzeGroupSize) and (len(group_j) >= min_analyzeGroupSize):
                            # Log output text
                            if not summary_header:
                                print('\t' + str(trial))
                                summary_header = True
                            groupDescrip_nocnt = str(groups_ALL[i]).upper() + ' vs ' + str(groups_ALL[j]).upper()
                            groupDescrip       = str(groups_ALL[i]).upper() + '(' + str(len(group_i_forJ)) + ') vs ' + str(groups_ALL[j]).upper()+ '(' + str(len(group_j)) + ')'
                            groupDescrip      += ''.join([' '] * (20 - len(groupDescrip)))

                            # Get statistical results
                            result_stat, output_line = summarize.get_statistics( group_i_forJ, group_j, description=groupDescrip, pvalue_threshold=pvalue_threshold)

                            # Log output text
                            output_text.append(output_line)
                            if (result_stat['ranksum_p'] < pvalue_threshold) and (result_stat['kruskal_p'] < pvalue_threshold):
                                result_stat.update( {'metric': metric, 'trial': trial, 'groups_cnt': groupDescrip, 'groups': groupDescrip_nocnt})
                                if metric in title_mapper.keys():
                                    result_stat['metric'] = metric + ' (' + title_mapper[metric] + ')'
                                output_stat.append( result_stat)
                                
                        # Delete variables to assure no data leakage/referencing issues
                        del group_j
                        del group_i_forJ
                        del ages_use
                del group_i
                del ages_use_i


        #############################################
        ##     BEGIN TRIAL DIFFERENCE ANALYSIS     ##
        #############################################
        if include_trialDifferences:
            # Log output text
            output_text.append('\n\nTRIAL DIFFERENCES\n')
            summary_trialDiff = pd.DataFrame()

            # Create a new trial ID to pair data from same session
            summary_ALL['uniqueID'] = summary_ALL['subject'].astype(str) + '+' + summary_ALL['trial_index'].astype(str)

            # Iterate through all trial combinations
            for a in range(len(trials_ALL)):
                for b in range(a+1, len(trials_ALL)):
                    # Create new name for this difference trial
                    trial_difference = str(trials_ALL[a]) + '-MINUS-' + str(trials_ALL[b])
                    # Log Output
                    output_text.append('\t' + trial_difference)
                    summary_header = False
                    print('\t', trial_difference)

                    # Separate out the trials of interest
                    summary_ALL_trial_a = summary_ALL.loc[ (summary_ALL['trial'] == trials_ALL[a]) & (summary_ALL[metric].notna())]
                    summary_ALL_trial_b = summary_ALL.loc[ (summary_ALL['trial'] == trials_ALL[b]) & (summary_ALL[metric].notna())]
                    # Only keep subjects which have both trials
                    summary_ALL_trial_a = summary_ALL_trial_a[ summary_ALL_trial_a['uniqueID'].isin( summary_ALL_trial_b['uniqueID'])]
                    summary_ALL_trial_b = summary_ALL_trial_b[ summary_ALL_trial_b['uniqueID'].isin( summary_ALL_trial_a['uniqueID'])]

                    # Compare all the groups (group "i" and group "j") to one another
                    for i in range(len(groups_ALL)):
                        # Define group "i"
                        group_i_a  = summary_ALL_trial_a.loc[ summary_ALL_trial_a['group'] == groups_ALL[i], :].set_index('uniqueID')
                        group_i_b  = summary_ALL_trial_b.loc[ summary_ALL_trial_b['group'] == groups_ALL[i], :].set_index('uniqueID')
                        # Store the difference between the metrics in new dataframe to account for potential age/subject mismatch in summary_ALL
                        group_i    = pd.DataFrame({metric : (group_i_a[metric] - group_i_b[metric]), 'age' : summarize.age_combine((group_i_a, group_i_b)), 'subject' : summarize.subject_combine((group_i_a, group_i_b)), 'sessionID' : summarize.session_combine((group_i_a, group_i_b))})
                        ages_use_i = summarize.get_ageInclude( group_i, age_boundary)

                        # Plot correlations
                        if plot_correlations and (len(group_i) >= min_analyzeGroupSize):
                            for metadata in show_corrPlotsWith:
                                group_i[metadata] = group_i_a[metadata]
                            result_corr = summarize.get_correlations( group_i, metric, show_corrPlotsWith, trial_difference + ' ' + groups_ALL[i], title_mapper=title_mapper, normalize=False, pvalue_threshold=pvalue_threshold, path_output=path_out_corrDifferences, save_outputs=save_outputs, save_prefix=groups_ALL[i], show_plots=show_plots)
                            for  _, row in result_corr.iterrows():
                                temp = row.to_dict()
                                temp.update({'trial': trial, 'cnt': len(group_i), 'group': groups_ALL[i]})
                                output_corr.append(temp)

                        # Record the new difference metrics separately for boxplots
                        df_temp = group_i_a.loc[:,['subject','group']].copy()
                        df_temp['trial'] = trial_difference
                        df_temp [metric] = group_i[metric]
                        summary_trialDiff = pd.concat([summary_trialDiff, df_temp], ignore_index=True)

                        if calc_statistics and (len(group_i) >= min_analyzeGroupSize):
                            for j in range(i+1, len(groups_ALL)):
                                # Define group "j"
                                group_j_a = summary_ALL_trial_a.loc[ summary_ALL_trial_a['group'] == groups_ALL[j], :].set_index('uniqueID')
                                group_j_b = summary_ALL_trial_b.loc[ summary_ALL_trial_b['group'] == groups_ALL[j], :].set_index('uniqueID')
                                # Store the difference between the metrics in new dataframe to account for potential age/subject mismatch in summary_ALL
                                group_j   = pd.DataFrame({metric : (group_j_a[metric] - group_j_b[metric]), 'age' : summarize.age_combine((group_j_a, group_j_b)), 'subject' : summarize.subject_combine((group_j_a, group_j_b)), 'sessionID' : summarize.session_combine((group_j_a, group_j_b))})
                                ages_use  = summarize.get_ageInclude( group_j, age_boundary, existing_ages=ages_use_i)
                                
                                # Filter groups for age/publication/matched group constraings
                                if not use_staticGroups:
                                    group_i_forJ = group_i.loc[ group_i['age'].isin(ages_use), metric].dropna().to_numpy()
                                    group_j      = group_j.loc[ group_j['age'].isin(ages_use), metric].dropna().to_numpy()
                                else:
                                    if use_groupsForPub:
                                        if (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'AD'):
                                            group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                        elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PD'):
                                            group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                        elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PDM'):
                                            group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                        else:
                                            group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub[groups_ALL[i]]),metric].dropna().to_numpy()
                                        if (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'AD'):
                                            group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                        elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PD'):
                                            group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                        elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PDM'):
                                            group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                        else:
                                            group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub[groups_ALL[j]]),metric].dropna().to_numpy()
                                    else:
                                        group_i_forJ = group_i.loc[ :,metric].dropna().to_numpy()
                                        group_j      = group_j.loc[ :,metric].dropna().to_numpy()

                                if (len(group_i_forJ) >= min_analyzeGroupSize) and (len(group_j) >= min_analyzeGroupSize):
                                    # Log output text
                                    if not summary_header:
                                        print('\t' + str(trial_difference))
                                        summary_header = True
                                    groupDescrip_nocnt  = str(groups_ALL[i]).upper() + ' vs ' + str(groups_ALL[j]).upper()
                                    groupDescrip        = str(groups_ALL[i]).upper() + '(' + str(len(group_i_forJ)) + ') vs ' + str(groups_ALL[j]).upper()+ '(' + str(len(group_j)) + ')'
                                    groupDescrip       += ''.join([' '] * (20 - len(groupDescrip)))

                                    # Get statistical results
                                    result_stat, output_line = summarize.get_statistics( group_i_forJ, group_j, description=groupDescrip, pvalue_threshold=pvalue_threshold)
                                    
                                    # Log output text
                                    output_text.append( output_line)
                                    if (result_stat['ranksum_p'] < pvalue_threshold) and (result_stat['kruskal_p'] < pvalue_threshold):
                                        result_stat.update( {'metric': metric, 'trial': trial_difference, 'groups_cnt': groupDescrip, 'groups': groupDescrip_nocnt})
                                        if metric in title_mapper.keys():
                                            result_stat['metric'] = metric + ' (' + title_mapper[metric] + ')'
                                        output_stat.append( result_stat)

                                # Delete variables to assure no data leakage/referencing issues
                                del group_j
                                del group_i_forJ
                                del ages_use
                        del group_i
                        del ages_use_i

            if plot_boxplots:
                summarize.get_boxplots(summary_trialDiff, metric, group_order=boxplot_order_group, trial_order=boxplot_order_trialDiff, path_output=path_out_boxDifferences, save_outputs=save_outputs, show_plots=show_plots)

        
        #############################################
        ##       BEGIN STROOP METRIC ANALYSIS      ##
        #############################################
        if include_stroopDifferences:
            # Log output text
            output_text.append('\n\nSTROOP DIFFERENCES\n')
            summary_stroopDiff = pd.DataFrame()

            # Create a new trial ID to pair data from same session
            summary_ALL['uniqueID'] = summary_ALL['subject'].astype(str) + '+' + summary_ALL['trial_index'].astype(str)

            # Iterate through all the stroop trial combinations
            for stroopMetric, (a, b) in {'readingTime': ['word-naming', 'word-naming'], 'colorNaming': ['color-naming', 'word-naming'], 'interference': ['word-color-naming', 'color-naming']}.items():
                # Log output
                output_text.append('\t' + stroopMetric)
                summary_header = False
                print('\t', stroopMetric)

                # Separate out the trials of interest
                summary_ALL_trial_a = summary_ALL.loc[ (summary_ALL['trial'] == a) & (summary_ALL[metric].notna())]
                summary_ALL_trial_b = summary_ALL.loc[ (summary_ALL['trial'] == b) & (summary_ALL[metric].notna())]
                # Only keep subjects which have both trials
                summary_ALL_trial_a = summary_ALL_trial_a[ summary_ALL_trial_a['uniqueID'].isin( summary_ALL_trial_b['uniqueID'])]
                summary_ALL_trial_b = summary_ALL_trial_b[ summary_ALL_trial_b['uniqueID'].isin( summary_ALL_trial_a['uniqueID'])]

                # Compare all the groups (group "i" and group "j") to one another
                for i in range(len(groups_ALL)):
                    # Define group "i"
                    group_i_a = summary_ALL_trial_a.loc[ summary_ALL_trial_a['group'] == groups_ALL[i], :].set_index('uniqueID')
                    group_i_b = summary_ALL_trial_b.loc[ summary_ALL_trial_b['group'] == groups_ALL[i], :].set_index('uniqueID')
                    # Store the difference between the metrics in new dataframe to account for potential age/subject mismatch in summary_ALL
                    if stroopMetric == 'readingTime':
                        group_i = pd.DataFrame({metric : (group_i_a[metric]), 'age' : summarize.age_combine((group_i_a, group_i_b)), 'subject' : summarize.subject_combine((group_i_a, group_i_b)), 'sessionID' : summarize.session_combine((group_i_a, group_i_b))})
                    elif stroopMetric == 'colorNaming':
                        group_i = pd.DataFrame({metric : (group_i_a[metric] / (group_i_a[metric] + group_i_b[metric])), 'age' : summarize.age_combine((group_i_a, group_i_b)), 'subject' : summarize.subject_combine((group_i_a, group_i_b)), 'sessionID' : summarize.session_combine((group_i_a, group_i_b))})
                    elif stroopMetric == 'interference':
                        group_i = pd.DataFrame({metric : (group_i_a[metric] - group_i_b[metric]), 'age' : summarize.age_combine((group_i_a, group_i_b)), 'subject' : summarize.subject_combine((group_i_a, group_i_b)), 'sessionID' : summarize.session_combine((group_i_a, group_i_b))})
                    ages_use_i = summarize.get_ageInclude( group_i, age_boundary)

                    # Plot correlations
                    if plot_correlations and (len(group_i) >= min_analyzeGroupSize):
                        for metadata in show_corrPlotsWith:
                            group_i[metadata] = group_i_a[metadata]
                        result_corr = summarize.get_correlations( group_i, metric, show_corrPlotsWith, stroopMetric + ' ' + groups_ALL[i], title_mapper=title_mapper, normalize=False, pvalue_threshold=pvalue_threshold, path_output=path_out_corrStroop, save_outputs=save_outputs, save_prefix=groups_ALL[i], show_plots=show_plots)
                        for  _, row in result_corr.iterrows():
                            temp = row.to_dict()
                            temp.update({'trial': trial, 'cnt': len(group_i), 'group': groups_ALL[i]})
                            output_corr.append(temp)

                    # Record the new stroop metrics separately for boxplots
                    df_temp = group_i_a.loc[:,['subject','group']].copy()
                    df_temp['trial'] = stroopMetric
                    df_temp [metric] = group_i[metric]
                    summary_stroopDiff = summary_stroopDiff.append(df_temp, ignore_index=True)

                    if calc_statistics and (len(group_i) >= min_analyzeGroupSize):
                        for j in range(i+1, len(groups_ALL)):
                            # Define group "j"
                            group_j_a = summary_ALL_trial_a.loc[ summary_ALL_trial_a['group'] == groups_ALL[j], :].set_index('uniqueID')
                            group_j_b = summary_ALL_trial_b.loc[ summary_ALL_trial_b['group'] == groups_ALL[j], :].set_index('uniqueID')
                            # Store the difference between the metrics in new dataframe to account for potential age/subject mismatch in summary_ALL
                            if stroopMetric == 'readingTime':
                                group_j = pd.DataFrame({metric : group_j_a[metric], 'age' : summarize.age_combine((group_j_a, group_j_b)), 'subject' : summarize.subject_combine((group_j_a, group_j_b)), 'sessionID' : summarize.session_combine((group_j_a, group_j_b))})
                            elif stroopMetric == 'colorNaming':
                                group_j = pd.DataFrame({metric : (group_j_a[metric] / (group_j_a[metric] + group_j_b[metric])), 'age' : summarize.age_combine((group_j_a, group_j_b)), 'subject' : summarize.subject_combine((group_j_a, group_j_b)), 'sessionID' : summarize.session_combine((group_j_a, group_j_b))})
                            elif stroopMetric == 'interference':
                                group_j = pd.DataFrame({metric : (group_j_a[metric] - group_j_b[metric]), 'age' : summarize.age_combine((group_j_a, group_j_b)), 'subject' : summarize.subject_combine((group_j_a, group_j_b)), 'sessionID' : summarize.session_combine((group_j_a, group_j_b))})
                            ages_use = summarize.get_ageInclude( group_j, age_boundary, existing_ages=ages_use_i)

                            # Filter groups for age/publication/matched group constraings
                            if not use_staticGroups:
                                group_i_forJ = group_i.loc[ group_i['age'].isin(ages_use), metric].dropna().to_numpy()
                                group_j      = group_j.loc[ group_j['age'].isin(ages_use), metric].dropna().to_numpy()
                            else:
                                if use_groupsForPub:
                                    if (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'AD'):
                                        group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                    elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PD'):
                                        group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                    elif (groups_ALL[i] == 'CTL') and (groups_ALL[j] == 'PDM'):
                                        group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                    else:
                                        group_i_forJ = group_i.loc[ group_i['sessionID'].isin(group_pub[groups_ALL[i]]),metric].dropna().to_numpy()
                                    if (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'AD'):
                                        group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLa']       ),metric].dropna().to_numpy()
                                    elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PD'):
                                        group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLb']       ),metric].dropna().to_numpy()
                                    elif (groups_ALL[j] == 'CTL') and (groups_ALL[i] == 'PDM'):
                                        group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub['CTLc']       ),metric].dropna().to_numpy()
                                    else:
                                        group_j      = group_j.loc[ group_j['sessionID'].isin(group_pub[groups_ALL[j]]),metric].dropna().to_numpy()
                                else:
                                    group_i_forJ = group_i.loc[ :,metric].dropna().to_numpy()
                                    group_j      = group_j.loc[ :,metric].dropna().to_numpy()

                            if (len(group_i_forJ) >= min_analyzeGroupSize) and (len(group_j) >= min_analyzeGroupSize):
                                # Log output text
                                if not summary_header:
                                    print('\t' + str(stroopMetric))
                                    summary_header = True
                                groupDescrip_nocnt  = str(groups_ALL[i]).upper() + ' vs ' + str(groups_ALL[j]).upper()
                                groupDescrip        = str(groups_ALL[i]).upper() + '(' + str(len(group_i_forJ)) + ') vs ' + str(groups_ALL[j]).upper()+ '(' + str(len(group_j)) + ')'
                                groupDescrip       += ''.join([' '] * (20 - len(groupDescrip)))

                                # Get statisitical results
                                result_stat, output_line = summarize.get_statistics( group_i_forJ, group_j, description=groupDescrip, pvalue_threshold=pvalue_threshold)
                                
                                # Log output text
                                output_text.append( output_line)
                                if (result_stat['ranksum_p'] < pvalue_threshold) and (result_stat['kruskal_p'] < pvalue_threshold):
                                    result_stat.update( {'metric': metric, 'trial': stroopMetric, 'groups_cnt': groupDescrip, 'groups': groupDescrip_nocnt})
                                    if metric in title_mapper.keys():
                                        result_stat['metric'] = metric + ' (' + title_mapper[metric] + ')'
                                    output_stat.append( result_stat)
                            
                            # Delete variables to assure no data leakage/referencing issues
                            del group_j
                            del group_i_forJ
                            del ages_use
                    del group_i
                    del ages_use_i

            if plot_boxplots:
                summarize.get_boxplots(summary_stroopDiff, metric, group_order=boxplot_order_group, trial_order=boxplot_order_trial, path_output=path_out_boxStroop, save_outputs=save_outputs, show_plots=show_plots)
        
        if calc_statistics and save_outputs:
            with open( os.path.join( path_out_stats, metric+'.txt'), 'w') as f:
                f.writelines([line + '\n' for line in output_text])

    '''
        Save Summary of Results
    '''
    if plot_correlations and save_outputs:
        df_corr = pd.DataFrame(output_corr)
        df_corr.to_csv( os.path.join(path_output, folder_output_corr, subfolder_output, 'summary.csv'))
        df_corr_sigAD = df_corr[ (df_corr['pvalue'] < pvalue_threshold) & (df_corr['group'] == 'AD') & (df_corr['metric2'] == 'moca')].sort_values('metric1')
        df_corr_sigPD = df_corr[ (df_corr['pvalue'] < pvalue_threshold) & (df_corr['group'] == 'PD') & (df_corr['metric2'] == 'UPDRS3')].sort_values('metric1')
        df_corr_sigAD.to_csv( os.path.join(path_output, folder_output_corr, subfolder_output, 'summary_significantAD.csv'))
        df_corr_sigPD.to_csv( os.path.join(path_output, folder_output_corr, subfolder_output, 'summary_significantPD.csv'))

    if calc_statistics and save_outputs:
        df_stat = pd.DataFrame(output_stat)
        _, df_stat['ranksum_fdrcorrected'] = fdrcorrection( df_stat['ranksum_p'])
        _, df_stat['kruskal_fdrcorrected'] = fdrcorrection( df_stat['kruskal_p'])
        # df_stat_keep = df_stat[ (df_stat['ranksum_fdrcorrected'] < pvalue_threshold) & (df_stat['kruskal_fdrcorrected'] < pvalue_threshold)]
        df_stat_keep = df_stat

        df_stat_keep.to_csv( os.path.join( path_out_stats, '!summary.csv'))



    #############################################
    ##      BEGIN FEATURE PROFILE SUMMARY      ##
    #############################################
    if plot_featureProfile:
        # Store values to plot
        metric_plot = []
        metric_plotSection = []

        # Calculate each metric "score" to plot
        for metric in profile_metrics:
            # Log output
            print(metric)
            col_score = metric+'_profileScore'
            summary_ALL[col_score]  = 0
            # Plot these values
            metric_plot.append(col_score)
            metric_plotSection.append( category_mapper[metric])

            # Iterate through all the trials
            for trial in trials_ALL:
                # Separate out the trial of interest
                index_score = summary_ALL['trial'] == trial

                # Normalize all the data based on the control group
                index_norm  = (index_score & (summary_ALL['sessionID'].isin(group_pub['CTL'])))
                offset      = summary_ALL.loc[ index_norm, metric].dropna().mean()
                scale       = summary_ALL.loc[ index_norm, metric].dropna().std()
                
                # Calculate the "Impact" or scaling of this metric for this trial
                scoreImpact = 1
                # scoreImpact_meta = df_corr.loc[ (df_corr['metric1'] == metric) & (df_corr['metric2'] == reference) & (df_corr['group'] == group) & (df_corr['trial'] == trial)]
                # scoreImpact = 1 if scoreImpact_meta['corr'].values[0] < 0 else -1
                # if scoreImpact_meta['pvalue'].values[0] < pvalue_threshold:
                #     scoreImpact = scoreImpact_meta['corr'].values[0]
                # else:
                #     scoreImpact = 0

                # Save new score
                summary_ALL.loc[index_score, col_score]  = scoreImpact * summary_ALL.loc[ index_score, metric].subtract(offset).divide(scale)

        # Plot the calculated metric "score" for all subjects
        output_score = []
        for group in groups_ALL:
            for sub in sorted(summary_group['subject'].unique(), reverse=False):
                summary_subject = summary_ALL.loc[ (summary_ALL['group'] == group) & (summary_ALL['subject'] == sub), :]
                
                # Track score severity for this subject across sessions
                scores_subject = []
                yellow_subject = []
                red_subject    = []
                # Iterate through all the sessions
                for session in sorted(summary_subject['trial_index'].unique()):
                    # Log output
                    print('  ', sub, session)

                    # Get profile for that session
                    summary_session = summary_subject.loc[(summary_subject['trial_index'] == session)]
                    sessionScores = summarize.get_profile( summary_session, metric_plot, features_section=metric_plotSection, spider=True, trials=trials_ALL, description='_'.join(('profile', group, sub, str(session))), path_output=path_out_profSummary, save_outputs=save_outputs, show_plots=show_plots)

                    # Record the mean score magnitude for this sessions
                    scores_session = [np.nanmean( np.abs( s)) for s in sessionScores.values()]

                    # Track score severity for this subject
                    scores_subject.append( np.percentile(scores_session, 90)) # 90th percentile of all metrics
                    yellow_subject.append(np.sum( [ s > 2 for s in sessionScores.values()]))
                    red_subject.append(   np.sum( [ s > 4 for s in sessionScores.values()]))

                # Save max score across all sections
                score_final  = np.max( scores_subject)
                yellow_final = np.max(yellow_subject)
                red_final    = np.max(red_subject)
                output_score.append({'group':group,'subject':sub,'score':score_final,'yellow':yellow_final,'red':red_final})

        # Perform quick statistical test on the scores of each group
        df_score = pd.DataFrame(output_score)
        data_ctl = df_score.loc[ df_score['group'] == 'CTL', 'score'].to_numpy()
        data_pd  = df_score.loc[ df_score['group'] == 'PD',  'score'].to_numpy()
        data_ad  = df_score.loc[ df_score['group'] == 'AD',  'score'].to_numpy()
        data_pdm = df_score.loc[ df_score['group'] == 'PDM', 'score'].to_numpy()
        stat_pd, _  = summarize.get_statistics(data_ctl, data_pd)
        stat_ad, _  = summarize.get_statistics(data_ctl, data_ad)
        stat_pdm, _ = summarize.get_statistics(data_ctl, data_pdm)
        print('PD:  ', stat_pd['kruskal_p'])
        print('AD:  ', stat_ad['kruskal_p'])
        print('PDM: ', stat_pdm['kruskal_p'])

        # Display quick visualization of score deviation
        df_score = df_score.reset_index()
        # sns.kdeplot( data=df_score, x='score',hue='group',common_norm=False)
        sns.swarmplot( data=df_score, x='score',y='group')

        color_palette = [ (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]
        sns.catplot(x='score', y='group', data=df_score, kind='boxen', k_depth='full',  sharex=False, height=6.2, aspect=0.8, order=boxplot_order_group, palette=color_palette, linewidth=1, showfliers=False)
        plt.xlabel('TREVOR SCORE')
        plt.show()

        if save_outputs:
            plt.savefig( os.path.join(path_output, folder_output_profile, subfolder_output, 'trevorscore.png'))
            df_score.to_csv( os.path.join(path_output, folder_output_profile, subfolder_output, 'summary.csv'))

    if show_plots:
        plt.show()



if __name__ == '__main__':
    main()
    print('\n\nFin.')
