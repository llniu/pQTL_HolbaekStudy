import statsmodels.stats.multitest as multi
import pandas as pd
import numpy as np
import pingouin as pg
from tqdm import tqdm

def pg_ttest(data, group_col, group1, group2, fdr=0.05, value_col='MS signal [Log2]'):
    '''
    data: long data format with ProteinID as index, one column of protein levels, other columns of grouping.
    '''
    df = data.copy()
    proteins = data.index.unique()
    columns = pg.ttest(x=[1,2], y=[3,4]).columns
    scores = pd.DataFrame(columns=columns)
    for i in proteins:
        df_ttest = df.loc[i]
        x=df_ttest[df_ttest[group_col]==group1][value_col]
        y=df_ttest[df_ttest[group_col]==group2][value_col]
        difference = y.mean()-x.mean()
        result = pg.ttest(x=x, y=y)
        result['protein']=i
        result['difference']=difference
        scores=scores.append(result)
    scores=scores.assign(new_column=lambda x: -np.log10(scores['p-val']))
    scores=scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['p-val'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    scores = scores.set_index('protein')
    return scores

def homoscedasticity_pg (data, dv, group):
    """
    This is a wrapper of pingouin.homoscedasticity test.
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "group": Name of column containing the between factor.
    More refer to: https://pingouin-stats.org/generated/pingouin.homoscedasticity.html
    """
    columns = ['W', 'pval', 'equal_var']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_homoscedasticity = data.loc[i]
        homoscedasticity = pg.homoscedasticity(data=df_homoscedasticity, dv=dv, group=group)
        homoscedasticity['protein'] = i
        scores = scores.append(homoscedasticity, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['pval']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    return scores

def normality_pg (data, dv, group, method='shapiro'):
    """
    This is a wrapper of pingouin.normality test.
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "group": Grouping factor.
    More refer to: https://pingouin-stats.org/generated/pingouin.normality.html
    """
    columns = ['index', 'W', 'pval', 'normal']
    scores = pd.DataFrame(columns = columns)
    scores = []
    for i in list(set(data.index)):
        df_normality = data.loc[i]
        normality = pg.normality(data=df_normality, dv=dv, group=group, method=method).reset_index()
        scores = scores.append(normality, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['pval']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    return scores

def perform_linear_regression(data, dep_vars, covariates, fdr_method='indep'):
    """
    Performs linear regression on a given dataset for multiple dependent variables.

    This function iterates over a list of dependent variables (dep_vars), performing linear regression against specified covariates for each. It handles missing data, calculates the residuals, and applies FDR correction to the p-values. It also determines the direction of the relationships.

    Parameters:
    data (pd.DataFrame): The dataset containing dependent variables and covariates.
    dep_vars (list): A list of column names in 'data' to be treated as dependent variables.
    covariates (list): A list of column names in 'data' to be used as covariates in the linear regression.

    Returns:
    tuple: A tuple containing two elements:
        - pd.DataFrame: A DataFrame with regression statistics for each dependent variable.
        - dict: A dictionary of residuals for each dependent variable.
    """

    stats, residuals = [], {}
    for dep_var in tqdm(dep_vars):
        df = data[[dep_var] + covariates].dropna()
        lm = pg.linear_regression(X=df[covariates], y=df[dep_var], relimp=True)
        residuals[dep_var] = pd.Series(lm.residuals_, index=df.index)
        lm = lm.assign(dep_var=dep_var, nr_obs=df.shape[0], df_model=lm.df_model_, df_residual=lm.df_resid_)
        stats.append(lm)

    stats = pd.concat(stats)
    if fdr_method=='indep':
        reject, qvalue = multi.fdrcorrection(stats['pval'], alpha=0.05, method=fdr_method)
        stats = stats.assign(qvalue=qvalue, rejected=reject)
    elif fdr_method == 'bonferroni':
        bonferroni_thresh = 0.05/len(dep_vars)
        stats['rejected']=np.where(stats['pval']<bonferroni_thresh, True, False)
    stats['-Log10 P-value'] = -np.log10(stats['pval'])
    stats['direction'] = np.where(stats['coef'] > 0, 'pos', 'neg')
    stats.loc[~stats['rejected'], 'direction'] = 'not significant'

    return stats, residuals

# Example usage:
# stats, residuals = perform_linear_regression(data, dep_vars, covariates)

def logistic_regression_pg(data, dep_var_list, indep_var, covariates):
    """
    Wrapper of pingouin.linear_regression for multiple testing. 
    Parameters
    ----------
    data: pandas dataframe wide format with rows of observations/samples and columns of proteins and phenotypic traits.
    covariates: list of covariates
    ----------
    """
    scores = []
    df = data.copy()
    for dep_var in dep_var_list:
        df_test = df[[dep_var, indep_var]+covariates].dropna()
        X=df_test[[indep_var]+covariates]
        y=df_test[dep_var]
        lom = pg.linear_regression(X=X, y=y)
        lom['dep_var']=dep_var
        scores.append(lom)
    scores=pd.concat(scores)
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['pval'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    scores['-Log10 P-value'] = -np.log10(scores['pval'])
    scores['direction']=np.where(scores['coef']>0, 'pos', 'neg')
    scores.loc[scores['rejected']== False, 'direction']='not significant'    
    return(scores)