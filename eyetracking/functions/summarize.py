from scipy.stats import ranksums, kruskal, spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from math import floor, ceil
from copy import deepcopy
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import sys
import gc
import os



def get_statistics( group_i, group_j, description='', pvalue_threshold=0.05):
    '''Returns dictionary containing statistics about the statistical independence of two groups.

    Multiple tests are run, including:
        * Kruskal
        * Ranksums
        * T-test

    An ROC_AUC score is also calculated.

    Args:
        group_i (array): Array containing one data metric from one group
        group_j (array): Array containing one data metric from another group
        description (str, optional): A descriptor to be added to a line of text summarizing the results.
        pvalue_threshold (float, optional): A threshold which decides whether the line will be print to the screen.
    Returns
        Dictionary summarizing statistical test results.
        String containign a summary of the statisitical tests, along with the mean and standard deviation of each group's data.
    '''
    try:
        result_kruskal, pvalue_kruskal = kruskal(  group_i, group_j)
        result_ranksum, pvalue_ranksum = ranksums( group_i, group_j)
        result_ttest,   pvalue_ttest   = ttest_ind( group_i, group_j)
        roc_auc                        = roc_auc_score( [0] * len(group_i) + [1] * len(group_j), np.concatenate((group_i, group_j), axis=0), average='micro')
        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc

        output_line = '\t\t' + description + '  :  ranksum p = {:.3f}\tkruskal p = {:.3f}\troc_auc = {:.3f}\tu={:.1f}vs{:.1f}, std={:.2f}vs{:.2f}\t(ttest p={:.3f})'.format( pvalue_ranksum, pvalue_kruskal, roc_auc, np.mean(group_i), np.mean(group_j), np.std(group_i), np.std(group_j), pvalue_ttest)

        if (pvalue_kruskal < pvalue_threshold) and (pvalue_ranksum < pvalue_threshold):
            print(output_line)

    except Exception as e:
        result_kruskal = 0
        result_ranksum = 0
        result_ttest   = 0
        pvalue_kruskal = 0.5
        pvalue_ranksum = 0.5
        pvalue_ttest   = 0.5
        roc_auc        = 0.5
        output_line = '\t\t' + description + '  :  ERROR CALCULATING STATS - ' + str(e)

    return {'ranksum_raw': result_ranksum, 'kruskal_raw': result_kruskal, 'ttest_raw': result_ttest, 'ranksum_p': pvalue_ranksum, 'kruskal_p': pvalue_kruskal, 'ttest_p': pvalue_ttest, 'roc_auc': roc_auc, 'avg': '{:.1f}vs{:.1f}'.format(group_i.mean(), group_j.mean()), 'std': '{:.2f}vs{:.2f}'.format(group_i.std(), group_j.std())}, output_line



def get_boxplots( df_data, col_data, group_order=None, trial_order=None, plot_points=True, path_output='./', save_outputs=True, show_plots=False):
    '''Function to generate and save boxplots, using data from the column of a dataframe.

    Args:
        df_data (pandas dataframe):    Dataframe containing all of the data
        col_data (str):                Identifier for the column of data in the dataframe to use when generating boxplots.
        group_order (list, optional):  The order in which the groups should appear on the y axis.
        trial_order (list, optional):  The order in which the trials should appear in the x direction
        plot_points (bool, optional):  Whether to also plot the raw data points for each trial
        path_output (str, optional):   Path to the output folder to save outputs to.
        save_outputs (bool, optional): Whether to save outputs
        show_plots (bool, optional):   Whether to show the outputs after generation.
    
    Returns:
        None. A figure is generated.
    '''
    color_palette = sns.color_palette()
    color_palette = [ (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

    if group_order is not None:
        display_order = []
        for g in group_order:
            groups = list( df_data.loc[ df_data['group'].str.contains(g), 'group'].unique())
            if g == 'PD':
                groups.remove('PDM')
            display_order.extend( sorted( groups))
    else:
        display_order = None

    plt.figure()
    g = sns.catplot(x=col_data, y='group', col='trial', data=df_data, kind='boxen', order=display_order, col_order=trial_order, palette=color_palette, k_depth='full', linewidth=1, showfliers=False)
    # g = sns.catplot(x=col_data, y='group', col='trial', data=df_data, kind='box', sharex=False, height=6.2, aspect=0.8, order=display_order, col_order=trial_order, palette=color_palette, linewidth=1, showfliers=False)
    # g.set_axis_labels(fontsize=10)
    g.set_ylabels('')
    plt.tight_layout(pad=3)

    if save_outputs:
        plt.savefig(os.path.join(path_output, str(col_data)+'-box.png'))
    if not show_plots:
        plt.clf()
        plt.close()
        gc.collect()


    if plot_points:

        plt.figure()
        warnings.simplefilter('ignore', UserWarning)
        g = sns.catplot(x=col_data, y='group', col='trial', data=df_data, kind='swarm', order=display_order, palette=color_palette, linewidth=0.5, sharex=False)
        # g = sns.catplot(x=col_data, y='group', col='trial', data=df_data, kind='swarm', order=display_order, palette=color_palette, linewidth=0.5)
        # g.set_axis_labels(fontsize=10)
        plt.tight_layout(pad=3)
        
        if save_outputs:
            plt.savefig(os.path.join(path_output, str(col_data)+'-pts.png'))
        if not show_plots:
            plt.clf()
            plt.close()
            gc.collect()



def get_correlations( df_data, col_metric, metric_compare, description='', title_mapper=None, normalize=False, pvalue_threshold=0.05, path_output='./', save_outputs=True, save_prefix='', show_plots=False):
    '''Function to generate and record correlation characteristics between many metrics

    Args:
        df_data (pandas dataframe):    Pandas dataframe containing all of the data
        col_metric (str):              Identifier of the column in `df_data` containing the data metric of interest
        metric_compare (str or list):  Identifier or list of identifiers of columns in `df_data` to correlate with col_metric.
        description (str, optional):   Descriptor of the data, to be added to saved filenames for easy reference.
        title_mapper (dict, optional): Dictionary mapping metric names to interpretable titles.
        normalize (bool, optional):    Whether to normaliez the data of interest to have mean of 0 and stdev of 1
        pvalue_threshold (float, optional): threshold deciding which decides whether to generate a plot or skip. Set to 1 to plot everything.
        path_output (str, optional):   Path to the output folder to save outputs to.
        save_outputs (bool, optional): Whether to save outputs
        save_prefix (str, optional):   Prefix added to filename
        show_plots (bool, optional):   Whether to show the outputs after generation. 
    '''
    metric_compare_use = deepcopy(metric_compare)
    for m in metric_compare:
        if df_data[m].notna().sum() == 0:
            metric_compare_use.remove(m)

    # g = sns.PairGrid(df_data.reset_index(), hue='group', palette='husl', vars=list(metric_compare_use)+[col_metric])
    # g.map_lower(sns.scatterplot)
    # g.map_diag(sns.histplot)
    # g.map_upper(sns.kdeplot)

    # if save_outputs:
    #     plt.savefig(os.path.join(path_output, description + '_' + str(col_metric) + '_corr.png'))
    # if not show_plots:
    #     plt.clf()
    #     plt.close()
    #     gc.collect()
    # return

    corr_final = pd.DataFrame()
    for metric_comp in metric_compare_use:
        temp_df = pd.DataFrame()
        # Need to redo this every time, since different rows may get dropped
        data_corr = df_data[[col_metric,metric_comp]].astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        # try:
        #     data_corr = df_data[[col_metric,metric_comp]].astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        # except:
        #     data_corr = df_data[[col_metric,metric_comp]].dropna(axis=0)

        np.seterr(divide='ignore', invalid='ignore')
        from scipy.stats import SpearmanRConstantInputWarning
        warnings.simplefilter('ignore', SpearmanRConstantInputWarning)
        corr, pvalue = spearmanr( data_corr[col_metric], data_corr[metric_comp])

        if pvalue < pvalue_threshold:
            if (title_mapper is not None) and (col_metric in title_mapper.keys()):
                label = title_mapper[col_metric] + '\n ρ={:.2f}'.format(corr)
            else:
                label = str(col_metric) + '-VS-' + str(metric_comp) + '\n ρ={:.2f}'.format(corr)
            print('\t\t' + metric_comp + '\tρ={:.2f} (p-{:.3f})'.format(corr, pvalue))

            temp_df = pd.DataFrame()
            if normalize:
                temp_df[col_metric] = (data_corr[col_metric] - data_corr[col_metric].mean()) / data_corr[col_metric].std()
                temp_df[metric_comp] = (data_corr[metric_comp] - data_corr[metric_comp].mean()) / data_corr[metric_comp].std()
            else:
                temp_df[col_metric] = data_corr[col_metric]
                temp_df[metric_comp] = data_corr[metric_comp]

            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            warnings.simplefilter('ignore', ConvergenceWarning)

            fg = sns.lmplot(x=metric_comp, y=col_metric, data=temp_df, height=6.2, aspect=1.5, scatter=True, palette="husl", robust=True, scatter_kws={"s": 50})
            plt.title( description + '\n' + label + ' (p={:.4f}) '.format(pvalue))
            plt.tight_layout(h_pad=2)
            if metric_comp == 'moca':
                plt.xlabel('MoCA')
                # Round down/up to the nearest inc
                inc = 2
                moca_min, moca_max = fg.ax.get_xlim()
                rng_min = inc * ceil(moca_min/inc)
                rng_max = inc * floor(moca_max/inc)
                fg.set(xticks=range(rng_min, rng_max+1, inc))
                plt.xlim(moca_max,moca_min)
            if metric_comp == 'UPDRS3':
                plt.xlabel('UPDRS-III')
                # Round down/up to the nearest inc
                inc = 5
                moca_min, moca_max = fg.ax.get_xlim()
                rng_min = inc * ceil(moca_min/inc)
                rng_max = inc * floor(moca_max/inc)
                fg.set(xticks=range(rng_min, rng_max+1, inc))
                # plt.xlim(moca_max,moca_min)

            path_out_final = os.path.join(path_output, metric_comp)
            if not( os.path.exists( path_out_final)) or not( os.path.isdir(path_out_final)):
                os.mkdir(path_out_final)
            if save_outputs:
                # plt.savefig(os.path.join(path_out_final, col_metric + '_' + '-'.join(description.strip().split()) + '_p={:.3f}.png'.format(pvalue)))
                if len(save_prefix) > 0 and not( save_prefix.endswith('_')):
                    save_prefix += '_'
                plt.savefig(os.path.join(path_out_final, save_prefix + '_'.join([ col_metric, '-'.join(description.strip().split()), 'p={:.3f}'.format(pvalue), 'rho={:.3f}.png'.format(corr)])))
            if not show_plots:
                plt.clf()
                plt.close()
                gc.collect()
        
        name_metric1 = col_metric
        name_metric2 = metric_comp
        if (title_mapper is not None):
            if (col_metric in title_mapper.keys()):
                name_metric1 += ' (' + title_mapper[col_metric] + ')'
            if (metric_comp in title_mapper.keys()):
                name_metric2 += ' (' + title_mapper[metric_comp] + ')'
        temp_df    = pd.DataFrame({'metric1': name_metric1, 'metric2': name_metric2, 'corr': corr, 'pvalue': pvalue}, index=[0])
        corr_final = pd.concat([corr_final, temp_df])
    
    # plt.figure()
    # sns.lmplot(x=col_metric, y=metric_comp, hue="Biomarker", data=correlate_final, height=6.2, aspect=1.5, scatter=True, palette="husl", robust=True)

    # if save_outputs:
    #     plt.savefig(os.path.join(path_output, 'corr_' + str(col_metric) + '-VS-all.png'))
    # if not show_plots:
    #     plt.clf()
    #     plt.close()
    #     gc.collect()
    return corr_final


def get_profile(df_data, features, features_section=None, spider=False, trials=None, description='', path_output='./', save_outputs=True, save_prefix='', show_plots=False):
    '''Function to generate and record profile summary of many metrics.

    Args:
        df_data (pandas dataframe):        Pandas dataframe containing all of the data
        features (list):                   Columns in `df_data` containing features to scale and plot
        features_section (list, optional): List of unique values aligned with `features` designating the section grouping of each feature
        spider (bool, optional):           Whether to use spider format or bar graph format
        trials (list, optional):           Trials to include in output
        description (str, optional):       Descriptor of the data, to be added to saved filenames for easy reference.
        path_output (str, optional):       Path to the output folder to save outputs to.
        save_outputs (bool, optional):     Whether to save outputs
        save_prefix (str, optional):       Prefix added to filename
        show_plots (bool, optional):       Whether to show the outputs after generation. 
    '''
    if trials is None:
        trials = df_data['trial'].unique()
    scores = {}
    plt.figure(figsize=[8,3.5])
    for i, trial in enumerate( sorted(trials)):
        plot_values  = df_data.loc[(df_data['trial'] == trial), features].values.flatten()
        if features_section is None:
            features_section = np.ones_like(plot_values)
            features_section[:4] = 2

        if len(plot_values) > 0:
            scores[trial] = plot_values
            scoreTitle = np.nanmean( np.abs( plot_values))
            # try:
            if True:
                if spider:
                    plot_values = np.clip( np.abs(plot_values), 0, 6)
                    plot_values = np.concatenate((plot_values, [plot_values[0]]))
                    spider_labels = ['_'.join(feat.split('_')[:-1]) for feat in features]

                    ax = plt.subplot(1,len(trials),i+1,polar=True)
                    ax.set_theta_offset(np.pi/2+0.1)
                    polar_axis = np.linspace( 0, 2*np.pi, num=len(features)+1)
                    plt.fill_between(polar_axis, np.ones_like(plot_values)*4, np.ones_like(plot_values)*2, facecolor='orange', alpha=0.5)
                    plt.fill_between(polar_axis, np.ones_like(plot_values)*2, facecolor='g', alpha=0.5)
                    polar_idx = 0
                    for s, section in enumerate( np.unique(features_section)):
                        plot_section  = plot_values[np.where( np.array(features_section) == section)]
                        polar_section = polar_axis[polar_idx:polar_idx+len(plot_section)]
                        plt.fill_between(polar_section, np.ones_like(plot_section)*6, np.ones_like(plot_section)*4, facecolor='k', alpha=0.08)
                        plt.plot(polar_section, np.ones_like(plot_section)*6, 'k', linewidth=2, label=section)
                        # plt.plot(polar_section, np.ones_like(plot_section)*4, 'orange', linewidth=2)
                        # plt.plot(polar_section, np.ones_like(plot_section)*2, 'g', linewidth=2)
                        plt.rgrids( radii=[2,4,6], labels=[], alpha=0.5)
                        if np.median(polar_section) > np.pi:
                            plt.text(np.median(polar_section), 6.7, section, fontsize='x-small', horizontalalignment='left')
                        else:
                            plt.text(np.median(polar_section), 6.7, section, fontsize='x-small', horizontalalignment='right')

                        for i, (x, y) in enumerate( zip(polar_section, plot_section)):
                            plt.plot([x,x],[0,y], 'b', linewidth=2)
                        # plt.plot( polar_axis, spider_values)
                        # plt.fill_between(polar_axis, spider_values, 'b')
                        # plt.thetagrids(np.degrees(polar_axis), labels=[*spider_labels, spider_labels[0]])

                        polar_idx += len(plot_section)
                    # plt.yticks([])
                    plt.xticks([])
                    plt.xlabel('{:.2f}'.format(scoreTitle))
                    plt.title(trial, fontweight='bold')

                else:
                    plt.subplot(len(trials),1,i+1)
                    plt.barh( features, plot_values)
                    ax = plt.gca()
                    ax.grid(False)
                    ax.axvspan(-4,4,color='orange',alpha=0.5)
                    ax.axvspan(-2,2,color='g',alpha=0.5)
                    lim = plt.xlim()
                    l = max( 10, *lim)
                    plt.xlim((-1*l,l))
                    if i < (len(trials)-1):
                        ax.get_xaxis().set_visible(True)
                    else:
                        plt.xlabel('Standard Deviation From CTL')
                    plt.title(trial+'  {:.2f}'.format(scoreTitle), loc='left', fontweight='bold', ha='center')
            # except:
            #     print('ERROR Processing ', sub, trial)
            #     continue
    plt.suptitle(description)
    plt.tight_layout()
    
    if save_outputs:
        if len(save_prefix) > 0 and not( save_prefix.endswith('_')):
            save_prefix += '_'
        plt.savefig(os.path.join(path_output, save_prefix+description+'.png'), dpi=300)
    if not show_plots:
        plt.clf()
        plt.close()
        gc.collect()
    return scores



def make_subgroups( df_data, df_metadata, col_subgroup, thresholds=[30,40,50,60,70,99]):
    '''Create new groups based on a metadata category

    Args:
        df_data (pandas dataframe):     Dataframe containing all data of interest.
        df_metadata (pandas dataframe): Dataframe containing metadata to use when generating new groups.
        col_subgroup (str):             Identifier for the column in df_metadata containing the information used to create new subgroups
        thresholds (list, optional):    List of thresholds to use when separating subroups must be split based on numerical metrics.

    Returns:
        df_data with new 'group' column replaced with new groups.  A column named ['group_og'] is created to maintain the original group assignment.
    '''
    df_data['group_og'] = df_data['group'].copy()

    if not( col_subgroup in df_metadata.columns) and (('proc_'+col_subgroup) in df_metadata.columns):
        col_subgroup = 'proc_'+col_subgroup

    if col_subgroup == 'gender':
        for g, gender in zip([1,2], ['m','f']):
            subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == g, 'subject']
            for sub in subjects_group:
                df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + gender
    else:
        if thresholds is None:
            subgroup_all = df_metadata[col_subgroup].unique()
            for g in subgroup_all:
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == g, 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + str(g)
        elif isinstance( thresholds, str):
            for thresh in thresholds:
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == thresh, 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + str(thresh)
        else:
            for thresh in sorted( thresholds, reverse=True):
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup].astype('float') <= float(thresh), 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_le' + str(thresh)

    return df_data



def get_ageInclude( df_data, age_boundary, existing_ages=None):
    '''Gets unique ages that should be included in analysis.  
    If existing ages are passed, ages separated by more than `age_boundary` are removed.

    Args:
        df_data (pandas dataframe):     Dataframe containing all data of interest.
        age_boundary (float):           Maximum separation between ages to include.
        existing_ages (list, optional): List of existing ages from antoher trial. 
                                        Intended to be the same values returned by this function in a previous call.
    Returns:
        numpy array of unique ages to include in analysis
    '''
    ages_use = []
    for a in df_data.loc[:,'age'].unique():
        if not np.isnan(a):
            ages_use.extend( list( range( int(a-age_boundary), int(a+age_boundary+0.99))))
    ages_use = np.unique(ages_use)

    if existing_ages is None:
        return ages_use
    else:
        ages_existing = np.unique( existing_ages.tolist())
        ages_use = ages_use[ np.isin( ages_use,ages_existing)]
        ages_existing = ages_existing[ np.isin(ages_existing,ages_use)]
        ages_use = ages_use[ np.isin( ages_use,ages_existing)]
        
        return np.unique(ages_use)

def age_combine( iterable_of_df):
    '''Combines ages from two dataframes. 
    If a subject is identified with different ages, the mean is taken.

    Args:
        iterable_of_df (pandas dataframe): List of dataframes containing 'age' column to combine
    Returns:
        pandas dataframe containing merged ages
    '''
    add_df = [ df.loc[:,'age'] for df in iterable_of_df]
    return pd.concat( add_df, axis=1).mean(axis=1)

def session_combine( iterable_of_df):
    '''Combines subjects from two dataframes. 
    If a session is identified with different ages, the min age is taken.

    Args:
        iterable_of_df (pandas dataframe): List of dataframes containing 'age' column to combine
    Returns:
        pandas dataframe containing merged subjects
    '''
    add_df = [ df.loc[:,'sessionID'] for df in iterable_of_df]
    return pd.concat( add_df, axis=1).min(axis=1)

def subject_combine( iterable_of_df):
    '''Combines subjects from two dataframes. 
    If a subject is identified with different ages, the min age is taken.

    Args:
        iterable_of_df (pandas dataframe): List of dataframes containing 'age' column to combine
    Returns:
        pandas dataframe containing merged subjects
    '''
    add_df = [ df.loc[:,'subject'] for df in iterable_of_df]
    return pd.concat( add_df, axis=1).min(axis=1)
