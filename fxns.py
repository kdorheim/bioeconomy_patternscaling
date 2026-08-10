"""
Helper functions pulled from stitches that are helpful!
"""

import fsspec
import intake
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import os

def extrat_exp_ens(file_path:str):
    # Extract the basename
    base_name = os.path.basename(file_path)

    # Extract the experiment name
    if "historical" in base_name:
        exp_name = "historical"
    elif "ssp585" in base_name:
        exp_name = "ssp585"
    else:
        raise Exception("Something went wrong.")

    # Extract the ensemble member name
    match = re.search(r"r\d+i\d+p\d+f\d+", base_name)
    ens_name = match.group()

    out = pd.DataFrame({"experiment": [exp_name], "ensemble": [ens_name]})
    return out


def fetch_nc(zstore: str):
    """
    Extract data for a single file from Pangeo.

    :param zstore: The location of the CMIP6 data file on Pangeo.
    :type zstore: str
    :return: An xarray Dataset containing CMIP6 data downloaded from Pangeo.
    """
    print(f"Fetching: {zstore}")

    # Function to update the progress bar
    def update_progress_bar(mapper, bar):
        keys = list(mapper.keys())

        for key in keys:
            _ = mapper[key]  # Trigger the actual read
            bar.update(1)

    # Create a file system mapper
    mapper = fsspec.get_mapper(zstore)

    # Initialize the progress bar
    with tqdm(total=len(mapper.keys()), desc="Downloading file components: ") as bar:
        update_progress_bar(mapper, bar)

    # Open the dataset
    ds = xr.open_zarr(mapper)

    if hasattr(ds, "time"):
        ds.sortby("time")

    return ds


def get_xr_meta(ds):
    """
    Get the metadata information from an xarray dataset.

    :param ds: xarray dataset of CMIP data.
    :return: pandas DataFrame of MIP information.
    """
    v = ds.variable_id

    data = [
        {
            "variable": v,
            "experiment": ds.experiment_id,
            "units": ds[v].attrs["units"],
            "frequency": ds.attrs["frequency"],
            "ensemble": ds.attrs["variant_label"],
            "model": ds.source_id,
        }
    ]
    df = pd.DataFrame(data)

    return df


def get_lat_name(ds):
    """Get the name for the latitude values in an xarray dataset.

    This function searches for latitude coordinates in the dataset,
    which could be named either 'lat' or 'latitude'.

    :param ds: The dataset from which to retrieve the latitude coordinate name.
    :type ds: xarray.Dataset
    :returns: The name of the latitude variable.
    :rtype: str
    :raises RuntimeError: If no latitude coordinate is found in the dataset.
    """
    for lat_name in ["lat", "latitude"]:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")






def global_total(ds):
    """
    Calculate the weighted global total for a variable in an xarray dataset.

    :param ds: The xarray dataset of CMIP data.
    :type ds: xarray.Dataset
    :returns: The xarray dataset of the weighted global mean.
    :rtype: xarray.Dataset
    """
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {"time"}
    return (ds * weight).mean(other_dims)



def selstr(a, start, stop):
    """
    Select elements of a string from start to stop index.

    :param a: Array containing a string.
    :type a: str
    :param start: First character index to select.
    :type start: int
    :param stop: Last character index to select.
    :type stop: int
    :returns: Array of strings.
    :rtype: list
    """
    if type(a) not in [str]:
        raise TypeError("a: must be a single string")

    out = []
    for i in range(start, stop):
        out.append(a[i])
    out = "".join(out)
    return out

def get_lat_name(ds):
    """Get the name for the latitude values in an xarray dataset.

    This function searches for latitude coordinates in the dataset,
    which could be named either 'lat' or 'latitude'.

    :param ds: The dataset from which to retrieve the latitude coordinate name.
    :type ds: xarray.Dataset
    :returns: The name of the latitude variable.
    :rtype: str
    :raises RuntimeError: If no latitude coordinate is found in the dataset.
    """
    for lat_name in ["lat", "latitude"]:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")


def subset_join(df1, df2):
    """
    Join two pandas data frames into a single data frame and subset the df1

    :param df1: First pandas DataFrame.
    :param df2: Second pandas DataFrame.
    :return: A single pandas DataFrame resulting from the joining of df1 and df2.
    """

    incommon = df1.columns.intersection(df2.columns).tolist()
    if len(incommon) == 0:
        raise TypeError("a: df1 and df2 must have names in common.")

    rows_df1 = df1.shape[0]
    rows_df2 = df2.shape[0]

    if (rows_df1 > rows_df2):
        raise TypeError("b: df1 needs to be the more restrictive data frame.")

    df1["to_keep"] = 1
    subset_df = df1.merge(df2, on=incommon, how="inner")
    subset_df = subset_df[subset_df["to_keep"] == 1]
    out = subset_df.drop('to_keep', axis=1)

    return out


def combine_df(df1, df2):
    """
    Join two pandas data frames into a single data frame.

    :param df1: First pandas DataFrame.
    :param df2: Second pandas DataFrame.
    :return: A single pandas DataFrame resulting from the joining of df1 and df2.
    """
    incommon = df1.columns.intersection(df2.columns)
    if len(incommon) > 0:
        raise TypeError("a: df1 and df2 must have unique column names.")

    # Combine the two data frames with one another.
    df1["j"] = 1
    df2["j"] = 1
    out = df1.merge(df2)
    out = out.drop(columns="j")

    return out


