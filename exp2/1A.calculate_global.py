# Process the carbon data files so that the global values account for the missing
# gridcells.

# 0. Set Up -----------------------------------------------------------------------------------------------------------
import fxns
import numpy as np
import pandas as pd
import os
import xarray as xr

# Prep where to write the processed data out to
DATA_DIR = "./exp2/data/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR+"field", exist_ok=True)
os.makedirs(DATA_DIR+"global", exist_ok=True)
os.makedirs(DATA_DIR+"beta", exist_ok=True)
os.makedirs(DATA_DIR+"intercept", exist_ok=True)
os.makedirs(DATA_DIR+"yhat", exist_ok=True)
os.makedirs(DATA_DIR+"error_metrics", exist_ok=True)

# Read in the data frame with the files to process, we can
# either pull directly from the online archive or
# from the locally loaded versions.
LOCAL = True

if LOCAL:
    to_process = pd.read_csv('./local_npp_rh_to_process.csv')
else:
    to_process = pd.read_csv('./npp_rh_to_process.csv')


# The options are 5, 10, 20, 50
MASK_LEVEL = 50


# 1. Training  ---------------------------------------------------------------------------------------------------------
# Add meta data information to the files to process
meta_data = to_process["zstore"].apply(fxns.extrat_exp_ens)
meta_data = pd.concat(meta_data.to_list(), ignore_index=True)
to_process_full = pd.concat([to_process, meta_data], axis=1)

for index, row in to_process_full.iterrows():

    data_path = to_process_full["zstore"][index]

    # Determine the tag name, this is how
    tag = os.path.basename(data_path).replace("gs://", "")
    tag = tag.replace("/", "_")
    tag = tag.replace("cmip6_CMIP6_CMIP_", "")
    tag = tag.replace("cmip6_CMIP6_ScenarioMIP_", "")

    if LOCAL:
        tag = tag.replace("._raw-data_data_", "")
        tag = tag.replace("_.nc", "")
        tag = tag.replace(".nc", "")

    # Load the required netcdf files.
    if LOCAL:
        d = xr.open_dataset(data_path)
        cell_area = xr.open_dataset(to_process["areacella"][index])
        land_percent = xr.open_dataset(to_process["sftlf"][index])
    else:
        d = fxns.fetch_nc(data_path)
        cell_area = fxns.fetch_nc(to_process["areacella"][index])
        land_percent = fxns.fetch_nc(to_process["sftlf"][index])


    # Extract variable information
    VAR = d.variable_id
    all_attrs = d.attrs
    original_units = d[VAR].units
    keys_to_keep = ["variable_id", "variant_label", "experiment_id", "frequency", "grid_label", "source_id"]
    meta_data = {k: all_attrs[k] for k in keys_to_keep if k in all_attrs}
    variant_label = meta_data["variant_label"]

    # Find the mask file of interest.
    patterns = [variant_label, VAR, str(MASK_LEVEL)+".nc"]
    for filename in os.listdir(DATA_DIR + "/mask"):
        if all(p in filename for p in patterns):
            mask_file = filename

    mask_file_nc = DATA_DIR + "/mask/" + mask_file
    mask = xr.open_dataset(mask_file_nc)

    if VAR in ["npp", "rh"]:

        # The first thing we need to do is get the annual field which should be total
        # carbon content per year.
        # TODO right now we are assuming a 365 day calander across models we may want to update this.
        seconds_per_year = (365 * 24 * 60 * 60) / 12
        annual_monthly_rate = d.coarsen(time=12).sum() * seconds_per_year

        # Calculate the land cell area, make sure that the land cell area where
        # land is 0 assign it a na value so that
        land_cell_area = cell_area.areacella * land_percent.sftlf * (1 / 100)
        land_cell_area = land_cell_area.where(land_cell_area != 0, other=np.nan)

       # Calculate the total annual carbon. Update attributes to reflect the
        # new units.
        # TODO figure out a way to add the attribtue information!
        field = annual_monthly_rate * land_cell_area.values

        # Convert from kg of C to the Petagram of C which are the
        # unites I am used to thinking in/hector units.
        field = field * 1e-12

        # Apply the mask to filter out the low levels of the
        field = field * mask

        other_dims = set(field.dims) - {"time"}
        global_ts = field.sum(other_dims, skipna=True)

        global_ofile = DATA_DIR + "global/" + tag + "_global_" + str(MASK_LEVEL) + ".nc"
        global_ts.to_netcdf(global_ofile, mode='w')