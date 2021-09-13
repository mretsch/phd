import sys
from os.path import expanduser
home = expanduser("~")
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns


def best_model_threshold_by_roc(df, percentiles):
    # True and False positive rates
    tpr, fpr = [], []
    # markedness, accuracy, f1_score = [], [], []
    stats_list = []

    assert 0 <= percentiles.min()
    assert      percentiles.max() <= 100

    for perc in percentiles:
        # Highest decile of ROME and predictions serve as targets
        df['actual_target'] = df['rome']     .gt(df['rome']     .quantile(  90/100))
        df['predic_target'] = df['predicted'].gt(df['predicted'].quantile(perc/100))

        # confusion_matrix = pd.crosstab(df['actual_target'], df['predic_target'], rownames=['ROME'], colnames=['Predicted'])
        # print(confusion_matrix)
        # sns.heatmap(confusion_matrix, annot=True)
        # plt.show()

        # ConfusionMatrix has nice stats, but throws an error if one array has exclusively True or False
        if (len(df['actual_target'].unique()) > 1) and (len(df['predic_target'].unique()) > 1):
            cm = ConfusionMatrix(df['actual_target'].to_list(), df['predic_target'].to_list())
            tpr.append(cm.stats()['TPR'])
            fpr.append(cm.stats()['FPR'])

            # stats = [cm.stats()[key] for key in cm.stats()]
            # stats_list.append(stats)
            stats_list.append(cm.stats().values())
        else:
            df['actual_target'] = df['actual_target'].astype(int)
            df['predic_target'] = df['predic_target'].astype(int)
            cm = pd.crosstab(df['actual_target'], df['predic_target'], rownames=['ROME'], colnames=['Predicted'])

            # only True predictions
            if cm.columns.get_values() == 1:
                tpr.append(1)
                fpr.append(1) #(cm.loc[0] / cm[1].sum())  # cm.loc[0] are the False target which are predicted True
            # ony False predictions
            elif cm.columns.get_values() == 0:
                tpr.append(0)
                fpr.append(0)
            else:
                raise ValueError('Targets consist of only True or only False.')

    plt.plot([0] + fpr, [0] + tpr, c='r', ls='', marker='o', ms=0.5)
    # loop through each x,y pair
    for i, xy in enumerate(zip(fpr, tpr)):
        corr = 0.#-0.05 # adds a little correction to put annotation in marker's centrum
        plt.annotate(str(percentiles[i].astype(int)),  xy=(xy[0] + corr, xy[1] + corr), fontsize=2)
    plt.xlabel('False positive rate (False positives / Target negative)')
    plt.ylabel('True positive rate (True positives / Target positive)')
    plt.title('Predictions of ROME by KitchenSink-NN and varying threshold for high organisation.')
    plt.axes().set_aspect('equal')
    plt.savefig(home + '/Desktop/ROC', dpi=400, bbox_inches='tight')

    # Distance to point (0, 1)
    dist = np.sqrt((0 - np.array(fpr))**2 + (1 - np.array(tpr))**2)
    best_threshold = percentiles[dist.argmin()]

    stats = pd.DataFrame(stats_list)
    stats.rename(dict(zip(list(range(26)), cm.stats().keys())), axis='columns', inplace=True)
    stats.index.rename('Threshold', inplace=True)
    stats.columns.rename('Statistics', inplace=True)
    stats = stats[::-1].reset_index(drop=True)

    return best_threshold, stats


if __name__=='__main__':

    rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
    # rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
    model_path = '/Desktop/'
    predicted = xr.open_dataarray(home + model_path + 'predicted.nc')

    rome = rome.where(predicted)

    # seaborn likes to have 'tidy' or 'long-form' data in a pd-dataframe
    # https://seaborn.pydata.org/tutorial/regression.html#functions-to-draw-linear-regression-models
    df = pd.DataFrame()
    df['rome'] = rome.values
    df['predicted'] = predicted.values

    #########
    # Scatter
    #########

    # seaborn's displot don't has colorbar out of the box
    # sns.displot(x='rome', y='predicted', data=df)
    # sns.kdeplot(rome, predicted, zorder=0, n_levels=6, shade=True,
    #     cbar=True, shade_lowest=False, cmap='viridis')
    # plt.savefig(home+'/Desktop/rome_scatter.pdf', bbox_inches='tight')

    ##############
    # Hit and Miss
    ##############
    most_accurate_percentile, stats = best_model_threshold_by_roc(df, percentiles=np.arange(99., -1, -1))

    df['actual_class'], r_bins = pd.qcut(df['rome'],      10, labels=list(range(1, 11)), retbins=True)
    df['predic_class'] = pd.qcut(df['predicted'],         10, labels=list(range(1, 11)))
    # df['predic_class'] = pd.cut(df['predicted'], bins=r_bins, labels=list(range(1, 11)))

    cm = ConfusionMatrix(df['actual_class'].to_list(), df['predic_class'].to_list())
    cm.print_stats()
    statdict = cm.stats()
    cm_stats = statdict['class']

    matrix = cm.to_dataframe()
    matrix.index.rename('ROME decile', inplace=True)
    matrix.columns.rename('R$_\mathrm{NN}$ decile', inplace=True)

    plt.close()
    # sns.heatmap(matrix / (len(predicted)//10) , cmap='Greys', annot=matrix, fmt='d')
    ax = sns.heatmap(matrix / (len(predicted)//10) , annot=matrix, fmt='d', cmap='gray_r')
    ax.collections[0].colorbar.set_label("Percentage in decile [1]")
    plt.savefig(home+'/Desktop/conf_matrix', dpi=400, bbox_inches='tight')