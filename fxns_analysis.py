
import pandas as pd
import re
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

def get_meta_data(f):
    """
    Extract meta data from teh file name.

    :param f: the full processed file name
    :return: data frame of file meeta data infromation ie model name, ensemble member ect.
    """
    pattern = re.compile(
        r'(?P<institution>[^_]+)_'
        r'(?P<model>[^_]+)_'
        r'(?P<experiment>[^_]+)_'
        r'(?P<ensemble>[^_]+)_'
        r'(?P<frequency>[^_]+)_'
        r'(?P<variable>[^_]+)_'
        r'(?P<grid>[^_]+)_'
        r'(?P<version>v\d+)_'
        r'(?P<metric>[^.]+)\.nc'
    )

    match = pattern.search(Path(f).name)

    df = pd.DataFrame([match.groupdict()])
    df = df.drop(columns=["version", "grid", "frequency"])

    return df


def get_meta_data2(f):
    """
    Extract meta data from teh file name.

    :param f: the full processed file name
    :return: data frame of file meeta data infromation ie model name, ensemble member ect.
    """
    pattern = re.compile(
        r'(?P<variable>[^.]+)_'
        r'(?P<ensemble>[^_]+)_'
        r'(?P<metric>[^.]+)\.nc'
    )

    match = pattern.search(Path(f).name)

    df = pd.DataFrame([match.groupdict()])

    return df


def get_meta_data3(f):
    """
    Extract meta data from teh file name.

    :param f: the full processed file name
    :return: data frame of file meeta data infromation ie model name, ensemble member ect.
    """
    pattern = re.compile(
        r'(?P<variable>[^.]+)_'
        r'(?P<ensemble>[^_]+)_'
        r'(?P<metric>[^.]+)_'
        r'(?P<mask>[^.]+)\.nc'
    )

    match = pattern.search(Path(f).name)

    df = pd.DataFrame([match.groupdict()])

    return df



def get_hist_data3(files):
    """
    Get the data for easy histogram plotting

    :param files: array of file names
    :return: data frame of results that are ready to be ploted in a histogram...
    """
    # create an empty data frame to store the results in
    metric_df = pd.DataFrame()

    for f in files:
        #print(f)
        meta_data = get_meta_data3(f)
        ds = xr.open_dataset(f)
        # Problem is this going to be influences by all the 0 from the ocean??? may be we need a better masking step
        # for when land area = 0....
        data_values = ds[meta_data.variable[0]].values.flatten()

        N = data_values.shape[0]
        df_expanded = pd.concat([meta_data] * N, ignore_index=True)
        df_expanded["value"] = data_values

        metric_df = pd.concat([metric_df, df_expanded], axis=0)

    metric_df = metric_df.reset_index()

    return metric_df

def get_hist_data2(files):
    """
    Get the data for easy histogram plotting

    :param files: array of file names
    :return: data frame of results that are ready to be ploted in a histogram...
    """
    # create an empty data frame to store the results in
    metric_df = pd.DataFrame()

    for f in files:
        #print(f)
        meta_data = get_meta_data2(f)
        ds = xr.open_dataset(f)
        ds = ds.where(ds.lat > -60)

        VAR = list(ds.data_vars)[0]


        # Problem is this going to be influences by all the 0 from the ocean??? may be we need a better masking step
        # for when land area = 0....
        data_values = ds[VAR].values.flatten()

        N = data_values.shape[0]
        df_expanded = pd.concat([meta_data] * N, ignore_index=True)
        df_expanded["value"] = data_values

        metric_df = pd.concat([metric_df, df_expanded], axis=0)

    metric_df = metric_df.reset_index()

    return metric_df


def get_hist_data(files):
    """
    Get the data for easy histogram plotting

    :param files: array of file names
    :return: data frame of results that are ready to be ploted in a histogram...
    """
    # create an empty data frame to store the results in
    metric_df = pd.DataFrame()

    for f in files:
        #print(f)
        meta_data = get_meta_data(f)
        ds = xr.open_dataset(f)
        # Problem is this going to be influences by all the 0 from the ocean??? may be we need a better masking step
        # for when land area = 0....
        data_values = ds[meta_data.variable[0]].values.flatten()

        N = data_values.shape[0]
        df_expanded = pd.concat([meta_data] * N, ignore_index=True)
        df_expanded["value"] = data_values

        metric_df = pd.concat([metric_df, df_expanded], axis=0)

    metric_df = metric_df.reset_index()

    return metric_df

def get_hist_data_antiantartica(files):
    """
    Get the data for easy histogram plotting

    :param files: array of file names
    :return: data frame of results that are ready to be ploted in a histogram...
    """
    # create an empty data frame to store the results in
    metric_df = pd.DataFrame()

    for f in files:
        #print(f)
        meta_data = get_meta_data(f)
        ds = xr.open_dataset(f)
        ds = ds.where(ds.lat > -60)

        # Problem is this going to be influences by all the 0 from the ocean??? may be we need a better masking step
        # for when land area = 0....
        data_values = ds[meta_data.variable[0]].values.flatten()

        N = data_values.shape[0]
        df_expanded = pd.concat([meta_data] * N, ignore_index=True)
        df_expanded["value"] = data_values

        metric_df = pd.concat([metric_df, df_expanded], axis=0)

    metric_df = metric_df.reset_index()

    return metric_df



def quick_plot_error_maps(files):
    """
    Quickly make a map of the error metrics

    :param files: array of file names
    :return: maps
    """
    if len(files) != 4:
        raise Exception("need to plot 4 data files")

    # Suppose you have four DataArrays:
    # da1, da2, da3, da4
    da1 = xr.open_dataset(files[0])
    da2 = xr.open_dataset(files[1])
    da3 = xr.open_dataset(files[2])
    da4 = xr.open_dataset(files[3])

    meta_data = pd.DataFrame()
    for f in files:
        single_meta_data = get_meta_data(f)
        meta_data = pd.concat([meta_data, single_meta_data], axis=0)
    meta_data = meta_data.reset_index()
    meta_data["exp_ens"] = meta_data["experiment"] + " " + meta_data["ensemble"]

    # Make sure that we are only looking at a single variable & metric
    if len(meta_data["variable"].unique()) != 1:
        raise Exception("only one variable can be plotted at a time")

    if len(meta_data["metric"].unique()) != 1:
        raise Exception("only one metric can be plotted at a time")

    variable_metric_title = meta_data["variable"].unique() + " " + meta_data["metric"].unique()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        constrained_layout=True
    )

    # Flatten axes for easy looping
    ax_list = axes.ravel()
    data_list = [da1, da2, da3, da4]
    titles = meta_data["exp_ens"]

#    fig, axes = plt.subplots(
#        2, 2,
#        figsize=(12, 10),
#        constrained_layout=True
#    )
    VAR = meta_data["variable"].unique()[0]
    ax_list = axes.ravel()

    for ax, da, title in zip(ax_list, data_list, titles):
        da[VAR].plot(
            ax=ax,
            cmap="viridis",  # choose your colormap
            add_colorbar=True
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")  # or whatever your x-axis is
        ax.set_ylabel(meta_data["metric"].unique()[0])  # or whatever your x-axis is
    fig.suptitle(variable_metric_title, fontsize=18, y=1.00)
    plt.show()

