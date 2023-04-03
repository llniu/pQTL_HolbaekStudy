import statsmodels.stats.multitest as multi
import pandas as pd
import numpy as np
import pingouin as pg
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
    for i in list(set(data.index)):
        df_normality = data.loc[i]
        normality = pg.normality(data=df_normality, dv=dv, group=group, method=method).reset_index()
        normality['protein'] = i
        scores = scores.append(normality, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['pval']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    return scores

def linear_regression_pg(data, dep_var_list, indep_var, covariates):
    """
    Wrapper of pingouin.linear_regression for multiple testing. 
    Parameters
    ----------
    data: pandas dataframe wide format with rows of observations/samples and columns of proteins and phenotypic traits.
    covariates: list of covariates
    ----------
    """
    scores = []
    dict_residuals = {}
    df = data.copy()
    for dep_var in dep_var_list:
        df_test = df[[dep_var, indep_var]+covariates].dropna()
        X=df_test[[indep_var]+covariates]
        y=df_test[dep_var]
        lm = pg.linear_regression(X=X, y=y,relimp=True )
        lm['dep_var']=dep_var
        residuals = pd.Series(lm.residuals_, index=df_test.index)
        scores.append(lm)
        dict_residuals[dep_var]=residuals
    scores=pd.concat(scores)
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['pval'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    scores['-Log10 P-value'] = -np.log10(scores['pval'])
    scores['direction']=np.where(scores['coef']>0, 'pos', 'neg')
    scores.loc[scores.rejected == False, 'direction']='not significant'  
    residuals_df = pd.DataFrame.from_dict(dict_residuals)
    return(scores, residuals_df)

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