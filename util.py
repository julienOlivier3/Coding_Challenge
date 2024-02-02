import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_describe(df, columns, hue=None, bins=50, bw_method=0.1, size=(22, 24), ncols=4):
    """Function to visualize columns in df. Visualization type depends on data type of the column.
    
    Arguments
    ---------
    df : pandas.DataFrame
        Dataframe whose columns shall be visualized.
    columns : list
        Subset of columns which shall be considered only.
    hue: str
        Column according to which single visualization shall be grouped.
    bins : int
        Number of bins for the histogram plots.
    bw_method : float
        method for determining the smoothing bandwidth to use.
    size: tuple
        Size of the resulting grid.
    nclos: int
        Number of columns in the resulting grid.
            

    Returns
    -------
    Visualization of each variable in columns as barplot or histogram.

    """

    # Reduce df to relevant columns
    df = df[columns]
    
    # Calculate the number of rows and columns for the grid
    num_cols = len(df.columns)
    num_rows = int(num_cols / ncols) + (num_cols % ncols)
    
    # Create the subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=size)
    
    # Flatten the axes array for easier iteration
    axes = axes.flatten()
    
    # Iterate over each column and plot accordingly
    for i, column in enumerate(df.columns):
        ax = axes[i]
        # Barplots for categorical features or integers with few distinct values
        if (df[column].dtype == 'int64' and df[column].value_counts().shape[0] < 10) or df[column].dtype == 'object':
            if hue==None or hue==column:
                df[column].value_counts().sort_index().plot(kind='bar', ax=ax, ylabel='Count', xlabel='', title=column)
            else:
                temp = df[[column, hue]].groupby(hue).value_counts(normalize=True).sort_index().to_frame().reset_index()
                temp[hue] = temp[hue].astype(str)
                p = sns.barplot(temp, x=column, y='proportion', hue=hue, errorbar=None, ax=ax)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Proportion')

        # Histograms for floats or integers with many distinct values
        elif (df[column].dtype == 'int64' and df[column].value_counts().shape[0] >= 10) or df[column].dtype == 'float64':
            if hue==None:
                df[column].plot(kind='hist', ax=ax, bins=bins, title=column)
            else:
                hue_groups = np.sort(df[hue].unique())
                for hue_group in hue_groups:
                    p = sns.kdeplot(data=df[df[hue] == hue_group], x=column, fill=True, label=hue_group, ax=ax, bw_method=bw_method)
                # Add title and labels
                p.set_title(column)
                p.set_xlabel('')
                p.set_ylabel('Density')
                p.legend(title=hue)
                                
        # For all other data types pass
        else:
            pass