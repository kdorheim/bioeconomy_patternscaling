# Apply the masks to the error metrics so we can try and take a look a how much these lower values
# are driving some of the biggest errors. This does assume that the experiment 1 has been run and also
# that the make mask files have been run.

# 0. Set Up -----------------------------------------------------------------------------------------------------------
import fxns
import numpy as np
import pandas as pd
import os
import xarray as xr
import re

# Define all the directories that will be used in this script.
MASK_DIR = "./exp2/data/mask/"
IN_ERROR_DIR = "./exp1/data/error_metrics"
OUT_ERROR_DIR = "./exp2/data/error_metrics"

# Make the dir if needed
os.makedirs(OUT_ERROR_DIR, exist_ok=True)

VAR = "rh"

# 1. Helper Functions --------------------------------------------------------------------------------------------------
def multi_partition(text, separators):
    # Join patterns with | (OR)
    regex_pattern = '|'.join(map(re.escape, separators))
    # Lookahead ?= keeps the delimiter on the right part
    # Use ?<= to keep it on the left
    parts = re.split(f'({regex_pattern})', text, maxsplit=1)

    if len(parts) > 1:
        return parts[0], parts[1], parts[2]
    return parts[0], "", ""

# 1. Set Up -----------------------------------------------------------------------------------------------------------

# Find all the mask files that will be applied to the different error metrics....
all_mask_files = [
    os.path.join(MASK_DIR, f)
    for f in os.listdir(MASK_DIR)
    if f.endswith("nc")
]

# For a specific mask level, apply the mask file to the error metrics...
for mf in all_mask_files:
    print(mf)

    # Load the mask file that is going to be applied to the various error metrics.
    mask = xr.open_dataset(mf)

    # Extract the ensemble variant label
    before, sep, after = multi_partition(mf, ["historical_", "ssp585_"])
    ENS, sep, after = after.partition("_Lmon")

    # Determine the mask level
    match = re.search(r"(\d+)(?=\.nc$)", mf)
    if match:
        MASK_LEVEL = match.group(1)

    # Determine which variable is included in the
    VAR = None
    if "npp" in mf.lower():
        VAR = "npp"
    elif "rh" in mf.lower():
        VAR = "rh"

    # Now find all the files that we need to apply the masks to.
    all_error_files = [
        os.path.join(IN_ERROR_DIR, f)
        for f in os.listdir(IN_ERROR_DIR)
        if re.search(VAR, f)
           and re.search(ENS, f)
           and f.endswith(".nc")
    ]

    for err_f in all_error_files:
        print(err_f)
        err_d = xr.open_dataset(err_f)
        masked_err = err_d[VAR] * mask[VAR]

        # Save the masked error results.
        root, ext = os.path.splitext(os.path.basename(err_f))
        out_file_name = os.path.join(OUT_ERROR_DIR, root + '_' + MASK_LEVEL + '.nc')
        masked_err.to_netcdf(out_file_name, mode='w')
        # end of the error file for loop
    # end of the mask level loop