import statsmodels.stats.multitest as multi
import pingouin as pg
import numpy as np

def pairwise_correlation (data, columns_x, columns_y, covar, fdr = 0.05, r_level=0.3, method = 'pearson'):
    """
    This is a wrapper of the pingouin.pairwise_corr for multiple hypothesis testing, with fdr control by Benjamini-Hochberg.
    Wide data format (samples, features), with sample ID as index, pair-wise values as two columns.
    """
    df = data.copy()
    scores = pg.pairwise_corr(data=df, columns=[columns_x, columns_y], covar=covar, method=method)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['p-unc']))
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['p-unc'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    sig = (scores.rejected) & (abs(scores['r'])>r_level)
    scores['sig'] = sig
    scores = scores.set_index('X')
    scores = scores.sort_values(by='r', ascending = False)

    return scores