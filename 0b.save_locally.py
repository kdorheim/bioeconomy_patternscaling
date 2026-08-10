# Recently it seems that there has been a major slow down in fetching the data from pangeo and
# this script reduces that burden by downlaoding the required raw data files and saving them locally.


import fxns
import numpy as np
import pandas as pd
import os



# Prep where to write the different data files out to
RAWDATA_DIR = "./raw-data"
DATA_DIR = os.path.join(RAWDATA_DIR, "data")
sftlf_DIR = os.path.join(RAWDATA_DIR, "sftlf")
areacella_DIR = os.path.join(RAWDATA_DIR, "areacella")

os.makedirs(RAWDATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(sftlf_DIR, exist_ok=True)
os.makedirs(areacella_DIR, exist_ok=True)

def cleanup_file_name(path: str):

    "Helper functions that cleans up the file name from the pangeo archive to something that is easier to work with"

    tag = data_path.replace("gs://", "")
    tag = tag.replace("/", "_")
    tag = tag.replace("cmip6_CMIP6_CMIP_", "")
    tag = tag.replace("cmip6_CMIP6_ScenarioMIP_", "")

    return tag



# Read in the data frame with the files to process
to_process = pd.read_csv('./npp_rh_to_process.csv')
local_to_process = to_process.copy()


# For each row in the to process data frame download all the required files,
# note that this is conservative and if the file already exits it will not
# redownload them.
for index, row in to_process.iterrows():

    print(index)
    data_path = to_process["zstore"][index]
    data_file_name = os.path.join(DATA_DIR, cleanup_file_name(data_path) + ".nc")

    if not os.path.isfile(data_file_name):
        d = fxns.fetch_nc(data_path)
        d.to_netcdf(data_file_name)

    local_to_process.loc[index, "zstore"] = data_file_name


    areacella_path = to_process["areacella"][index]
    areacella_file_name = os.path.join(areacella_DIR, cleanup_file_name(areacella_path) + ".nc")

    print(index)
    if not os.path.isfile(areacella_file_name):
        d = fxns.fetch_nc(areacella_path)
        d.to_netcdf(areacella_file_name)

    local_to_process.loc[index, "areacella"] = areacella_file_name

    sftlf_path = to_process["sftlf"][index]
    sftlf_file_name = os.path.join(sftlf_DIR, cleanup_file_name(sftlf_path) + ".nc")

    print(index)
    if not os.path.isfile(sftlf_file_name):
        d = fxns.fetch_nc(sftlf_path)
        d.to_netcdf(sftlf_file_name)

    local_to_process.loc[index, "sftlf"] = sftlf_file_name




local_to_process.to_csv("local_npp_rh_to_process.csv", index=False)