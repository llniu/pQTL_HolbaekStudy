import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def pre_processing(data_long, 
                                   sampleID_prefix='200ng_', 
                                   sampleID_suffix='_2023'):
    df=data_long.copy()
    sample_ids = df['R.FileName'].str.split(sampleID_prefix).str[1].str.split(sampleID_suffix).str[0]
    df['Sample ID']=sample_ids
    
    df = df.rename({'PG.Genes':'Gene names', 'PG.ProteinAccessions':'Protein IDs'}, axis=1)
    df['Protein ID']=df['Protein IDs'].str.split(';').str[0]
    df['Gene name']=df['Gene names'].str.split(';').str[0]
    df['ProteinID_Genename'] = df['Protein ID'] + '_' + df['Gene name']
    
    return (df)

def calculate_data_completeness(dataframe):
    """
    Calculate the completeness of data in a DataFrame.

    This function computes the completeness of data by determining the number of non-missing
    values (proteins) for each sample in the given DataFrame. It returns a DataFrame with the 
    number of present data points, the percentage completeness, and the rank of each protein.

    Args:
        dataframe (DataFrame): A wide-format DataFrame where rows represent proteins and columns represent samples.

    Returns:
        DataFrame: A DataFrame containing the number of non-missing values per protein ('Nr.PGs'), 
                   the percentage of completeness ('%Complete'), and the rank of each protein based on 
                   the number of non-missing values.

    Note:
        The input DataFrame should be in a wide format, with proteins as rows and samples as columns.
    """
    df = dataframe.copy()
    nr_proteins = df.shape[0]
    nr_samples = df.shape[1]
    df_comp = pd.DataFrame(df.count(axis=1), columns=['Nr.PGs']).sort_values(by='Nr.PGs', ascending=False)
    df_comp['%Complete'] = df_comp['Nr.PGs']/nr_samples
    df_comp['rank'] = np.arange(nr_proteins)
    return df_comp

def filter_data(dataframe, run_wise_thresh_dev=3, protein_wise_thresh_perc=0.6):
    """
    Filters the data based on run-wise and protein-wise thresholds.

    This function filters out runs and proteins from the data based on specified thresholds.
    Runs with a protein count less than (median - run_wise_thresh_dev*std) are dropped, and proteins
    with less than a certain percentage of valid values across all samples are also dropped.

    Args:
        dataframe (DataFrame): A wide-format DataFrame with proteins as rows and samples as columns.
        run_wise_thresh_dev (int, optional): The number of standard deviations below the median to use as the
                                             threshold for dropping runs. Defaults to 3.
        protein_wise_thresh_perc (float, optional): The minimum percentage of valid (non-missing) values required
                                                    for each protein across all samples. Defaults to 0.6.

    Returns:
        tuple: A tuple containing the list of retained proteins, the list of retained sample IDs, and the filtered DataFrame.

    Prints:
        Information about the number of samples and proteins dropped.

    Note:
        It is assumed that the input DataFrame is in a wide format, with proteins as rows and samples as columns.
    """
    
    # Drop runs
    df = dataframe.copy()
    threshold_nr_proteins_per_run = df.count().median() - run_wise_thresh_dev * df.count().std()
    runs_to_drop = df.columns[df.count() < threshold_nr_proteins_per_run]
    df_filtered = df.drop(runs_to_drop, axis=1)
    
    # Drop proteins
    threshold_nr_values_per_protein = df_filtered.shape[1] * protein_wise_thresh_perc
    proteins_to_drop = df_filtered.index[df_filtered.count(axis=1) < threshold_nr_values_per_protein]
    df_filtered = df_filtered.loc[~df_filtered.index.isin(proteins_to_drop)]
    proteins = df_filtered.T.columns.tolist()
    sample_ids = df_filtered.columns.tolist()
    
    print('Dropped {} samples and {} proteins'.format(len(runs_to_drop), len(proteins_to_drop)))
    return proteins, sample_ids, df_filtered

    
def summarize_filtered_data(data_filtered):
    """
    Summarizes the missing data and shape of the filtered data.

    This function calculates the overall percent of missing values in the filtered data and the
    percent of missing values per protein. It prints out a summary including the overall missing data percentage
    and the shape of the filtered data.

    Args:
        data_filtered (DataFrame): A DataFrame representing the filtered data, typically after applying some cleaning
                                   or reduction techniques.

    Prints:
        - The overall percentage of missing data in the filtered dataset.
        - The shape of the filtered dataset (number of proteins and samples).

    Returns:
        tuple: A tuple containing the overall missing value percentage and the per-protein missing value percentage.
    """
    # Calculate percent of missing values overall and per protein
    missing_value_percent = data_filtered.isna().mean().mean()
    missing_value_pct_perprotein = data_filtered.isnull().sum(axis=1) / data_filtered.shape[1]

    # Print summary information
    print(f'Missing data after filtering: {missing_value_percent:.2%}')
    print(f'Filtered data shape: {data_filtered.shape}')

def compute_pca(dataframe, n_components=10, random_state=2023):
    """
    Performs PCA on a given dataset.

    Args:
        dataframe (DataFrame): Input data with rows as samples and no missing values.
        n_components (int): Number of principal components. Defaults to 10.
        random_state (int): Seed for the random number generator. Defaults to 2023.

    Returns:
        pca (PCA): PCA object with the transformation.
        df_pc (DataFrame): Principal components.
        df_loadings (DataFrame): Loadings for each component.
    """
    # Standardize and perform PCA
    dataframe_scaled = preprocessing.StandardScaler().fit_transform(dataframe)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(dataframe_scaled)
    X_pca = pca.transform(dataframe_scaled)

    # Prepare DataFrame for principal components
    df_pc = pd.DataFrame(data=X_pca,
                         columns=['PC{}'.format(i) for i in range(1, n_components + 1)],
                         index=dataframe.index)

    # Calculate loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    df_loadings = pd.DataFrame(loadings,
                               columns=['PC{}'.format(i) for i in range(1, n_components + 1)],
                               index=dataframe.columns)

    return pca, df_pc, df_loadings

def generate_pca_plot(df_pc, pca, group_column='Grouping_batch', PCA_x=1, PCA_y=2, palette=None):
    """
    Generates a scatter plot for the specified principal components of a PCA-transformed dataset.

    Args:
        df_pc (DataFrame): DataFrame containing the PCA-transformed data.
        pca (PCA): The PCA object containing the variance ratio for labelling axes.
        group_column (str): Name of the column in df_pc to use for coloring groups. Defaults to 'Grouping_batch'.
        PCA_x (int): Index of the principal component for the x-axis. Defaults to 1.
        PCA_y (int): Index of the principal component for the y-axis. Defaults to 2.

    Returns:
        matplotlib.figure.Figure: The figure object containing the PCA plot.
    """
    # Setup labels and dimensions for the plot
    X, Y = f'PC{PCA_x}', f'PC{PCA_y}'
    explained_x = round(pca.explained_variance_ratio_[PCA_x-1]*100, 1)
    explained_y = round(pca.explained_variance_ratio_[PCA_y-1]*100, 1)

    # Create the scatter plot
    plt.figure(figsize=(4,4))
    scatter = sns.scatterplot(x=X, y=Y, data=df_pc, hue=group_column, alpha=1, s=10, palette=palette)
    scatter.set_xlabel(f'PC{PCA_x} ({explained_x}%)', fontsize=16)
    scatter.set_ylabel(f'PC{PCA_y} ({explained_y}%)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    # Adjust plot settings
    plt.rcParams['pdf.fonttype'] = 42

    # Return the figure object
    return plt.gcf()

def calculate_cv(dataframe, qa_samples):
    """
    Calculate the coefficient of variation (CV) for each protein across QA samples.

    Args:
        dataframe (pd.DataFrame): Input data with rows as proteins and columns as samples in wide format.
        qa_samples (list): List of QA sample columns on which the CV will be calculated.

    Returns:
        pd.DataFrame: A DataFrame containing the coefficient of variation and log2 protein abundance for each protein, sorted by CV.

    Note:
        The function assumes all values are positive and not zero (as log2 is applied).
    """
    # Filter dataframe for QA samples and calculate coefficient of variation
    df_qa = dataframe.copy()[list(set(qa_samples) & set(dataframe.columns))]
    coef_of_variation = lambda x: np.std(x) / np.mean(x)  # Define CV lambda function
    cvs = df_qa.apply(coef_of_variation, axis=1)
    log2_abundances = np.log2(df_qa.median(axis=1))  # Calculate median log2 abundance
    log10_abundances = np.log10(df_qa.median(axis=1))  # Calculate median log10 abundance

    # Create a new DataFrame with CV and log2 abundance
    df_cv = pd.DataFrame({
        'Coefficient of variation': cvs,
        'Protein abundance [Log2]': log2_abundances,
        'Protein abundance [Log10]': log10_abundances,
    }).sort_values(by='Protein abundance [Log2]', ascending = False)
    
    # Assign color based on CV threshold
    df_cv['color'] = np.where(df_cv['Coefficient of variation'] < 0.3, 'CV<30%', 'rest')
    df_cv['rank'] = np.arange(df_cv.shape[0])
    
    # Print median CV value
    print('Median CV: {:.2f}'.format(df_cv['Coefficient of variation'].median()))

    return df_cv

import matplotlib.pyplot as plt
import seaborn as sns

def generate_pca_plot1(df_pc, pca, group_column='Grouping_batch', PCA_x=1, PCA_y=2, palette=None, ax=None):
    """
    Generates a scatter plot for the specified principal components of a PCA-transformed dataset.

    Args:
        df_pc (DataFrame): DataFrame containing the PCA-transformed data.
        pca (PCA): The PCA object containing the variance ratio for labelling axes.
        group_column (str): Name of the column in df_pc to use for coloring groups. Defaults to 'Grouping_batch'.
        PCA_x (int): Index of the principal component for the x-axis. Defaults to 1.
        PCA_y (int): Index of the principal component for the y-axis. Defaults to 2.
        palette (dict or sequence): Mapping of hue levels to matplotlib colors. Defaults to None.
        ax (matplotlib.axes.Axes): The axes object to draw the plot onto, if None creates a new figure. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes object containing the PCA plot.
    """
    # Setup labels and dimensions for the plot
    X, Y = f'PC{PCA_x}', f'PC{PCA_y}'
    explained_x = round(pca.explained_variance_ratio_[PCA_x-1]*100, 1)
    explained_y = round(pca.explained_variance_ratio_[PCA_y-1]*100, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    # Create the scatter plot
    scatter = sns.scatterplot(x=X, y=Y, data=df_pc, hue=group_column, alpha=1, s=10, palette=palette, ax=ax)
    scatter.set_xlabel(f'PC{PCA_x} ({explained_x}%)', fontsize=16)
    scatter.set_ylabel(f'PC{PCA_y} ({explained_y}%)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    # Adjust plot settings if creating a new figure
    if ax is None:
        plt.rcParams['pdf.fonttype'] = 42

    # Return the axes object
    return ax
