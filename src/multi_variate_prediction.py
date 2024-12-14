import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from scipy.stats import pearsonr
import pingouin as pg


def get_data(data, outcome_col, predictors, stratify_col):
    if outcome_col!=stratify_col:
        df=data[predictors + [outcome_col, stratify_col]].dropna()
    else:
        df=data[predictors + [outcome_col]].dropna()
    X=df[predictors]
    y=df[outcome_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, 
                                                       stratify=df[stratify_col]
                                                       )
    X_train_train, X_train_validation, y_train_train, y_train_validation = train_test_split(X_train, y_train, 
                                                                                            test_size=0.3, 
                                                                                            random_state=42,
                                                                                           stratify=df[stratify_col].loc[X_train.index]
                                                                                           )
    return X_train_train, X_train_validation, X_test, y_train_train, y_train_validation, y_test

# Get correlation coefficients for features in training set
def get_features(outcome_col, predictors, stratify_col, data, top=50):
    X_train_train, X_train_validation, X_test, y_train_train, y_train_validation, y_test = get_data(data=data, outcome_col=outcome_col, 
                                                                                                   predictors=predictors, stratify_col=stratify_col)
    df_test = X_train_train.join(pd.DataFrame(y_train_train))
    corr_features=pg.pairwise_corr(df_test, columns=[predictors, outcome_col])
    corr_features['abs(r)']=abs(corr_features['r'])
    corr_features=corr_features.sort_values(by='abs(r)', ascending=False).set_index('X')

    mses = []
    for i in tqdm(np.arange(1, corr_features.shape[0]+1)):
        selected_features = corr_features.iloc[:i].index
        model = LinearRegression()
        model.fit(X_train_train[selected_features], y_train_train)
        y_pred_validation = model.predict(X_train_validation[selected_features])
        mse = mean_squared_error(y_train_validation, y_pred_validation)
        mses.append(mse)
    sme_vs_features = pd.DataFrame(mses, columns=['Squared mean error'])
    sme_vs_features['Nr. features']=np.arange(1, sme_vs_features.shape[0]+1)
    sme_vs_features['features'] = corr_features.index
    sme_vs_features.set_index('features', inplace=True)
    combined = corr_features.join(sme_vs_features)
    nr_of_features = int(combined[combined['Squared mean error']==combined['Squared mean error'].min()].iloc[0]['Nr. features'])
    nr_of_features = nr_of_features if nr_of_features < top else top
    features = combined.index[:nr_of_features]
    
    return (features, combined)

def get_prediction_score(outcome_col, data, predictors, features, stratify_col):
    X_train_train, X_train_validation, X_test, y_train_train, y_train_validation, y_test = get_data(outcome_col=outcome_col, data=data,
                                                                                                       predictors=predictors, stratify_col=stratify_col)
    model = LinearRegression()
    model.fit(X_train_train[features], y_train_train)
    coef = model.coef_
    y_pred_test = model.predict(X_test[features])
    corr, pvalue = pearsonr(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    pred_score = pd.DataFrame.from_dict({'Pearson r': corr.round(2), 'mean_squared_error':mse.round(1),
                                         'mean_absolute_error':mae.round(1)},
                                          orient='index', columns=[outcome_col])
    pred = pd.DataFrame(y_test)
    pred['y_pred'] = y_pred_test
    
    return(pred, pred_score, coef)