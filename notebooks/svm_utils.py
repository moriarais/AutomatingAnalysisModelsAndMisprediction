# data processing
import pandas as pd
import pickle

# data visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Processing functions
def getProcessedData(file_path: str):
    # read python dict back from the file
    with open(file_path, 'rb') as f:
        split_dict = pickle.load(f)

    X_train = split_dict["X_train"]
    X_test = split_dict["X_test"]
    y_train = split_dict["y_train"]
    y_test = split_dict["y_test"]
    return X_train, X_test, y_train, y_test

def getCSVData(url: str):
    dataset = pd.read_csv(url)
    return dataset



'''
Computes the correlation matrix.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson" (numeric), "spearman" (categorical), "kendall"
    :param negative: bool - if False it takes the absolute values of correlation
    :param lst_filters: list - filter rows to show
    :param annotation: logic - plot setting
'''
def corr_matrix(dtf, method="pearson", negative=True, lst_filters=[], annotation=True, figsize=(10,5)):    
    ## factorize
    dtf_corr = dtf.copy()
    for col in dtf_corr.columns:
        if dtf_corr[col].dtype == "O":
            print("--- WARNING: Factorizing", dtf_corr[col].nunique(),"labels of", col, "---")
            dtf_corr[col] = dtf_corr[col].factorize(sort=True)[0]
    ## corr matrix
    dtf_corr = dtf_corr.corr(method=method) if len(lst_filters) == 0 else dtf_corr.corr(method=method).loc[lst_filters]
    dtf_corr = dtf_corr if negative is True else dtf_corr.abs()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dtf_corr, annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    # set the y-axis labels vertically
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    return dtf_corr