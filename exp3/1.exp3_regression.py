# Process train and perform the linear regressions CO2 driven only. This will
# reuse some of the processed data from the exp0  (field nc files).
# 0. Set Up --------------------------------------------------------------------------
import re
import numpy as np
import pandas as pd
import os
import xarray as xr


# Prep where to write the processed data out to
DATA_DIR = "./exp3/data/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR+"beta", exist_ok=True)
os.makedirs(DATA_DIR+"intercept", exist_ok=True)
os.makedirs(DATA_DIR+"yhat", exist_ok=True)
os.makedirs(DATA_DIR+"error_metrics", exist_ok=True)


# 1. Main Chunk --------------------------------------------------------------------------

# Right now we are only training on the historical files, we will leave the future ones before.
historical_files = [
    os.path.join("exp0/data/field", f)
    for f in os.listdir("exp0/data/field")
    if f.endswith("nc")
    and re.search("historical", f)
]

# For each of the historical files, train linear model
for nc in historical_files:
    print(nc)
    base = os.path.basename(nc)
    root = base.replace("_field.nc", "")

    # load the netcdf file
    field = xr.open_dataset(nc)
    VAR = field.attrs['variable_id']
    ENS = field.attrs['variant_label']
    years = [t.year for t in field.time.values]

    # Read in the CO2 data, this is what is going to be used to drive the scenarios....
    co2_full_df = pd.read_csv("raw-data/co2/co2_ssp585.csv")
    historical_co2 = co2_full_df[co2_full_df["year"].isin(years)]
    co2_values = np.array(historical_co2['value'])

    # update the time element with the historical co2 values so that is
    # the explanatory variable used in the pattern scaling.
    field_with_global = field.copy()
    field_with_global = field_with_global.assign_coords(
        co2=("time", co2_values)
    )
    reg = field_with_global.polyfit(dim="co2", deg=1)
    reg = reg.rename({
        VAR+"_polyfit_coefficients": "coeff",
    })
    beta = reg.coeff.sel(degree=1)
    intercept = reg.coeff.sel(degree=0)

    # Save a copy of the slope and intercept
    beta.to_netcdf(DATA_DIR + "beta/" + root + "_beta.nc", mode='w')
    intercept.to_netcdf(DATA_DIR + "intercept/" + root + "_intercept.nc", mode='w')

    # First we need to load the future ssp585 data to use as our comparisons.
    ssp585_file = [
        os.path.join("exp0/data/field", f)
        for f in os.listdir("exp0/data/field")
        if f.endswith("nc")
           and re.search("ssp585", f)
           and re.search(VAR, f)
           and re.search(ENS, f)
    ][0]
    ssp585_data = xr.open_dataset(ssp585_file)
    og_time = ssp585_data.time.values

    # Update the root file name to reflect the scenario we are predicting.
    root = root.replace("historical", "ssp585")

    # Now we want to USE the slope and interecepts to predict future values and do the QAQC
    # analysis and also to construct the future values.
    # First get the future values for co2.
    future_co2 = co2_full_df[co2_full_df["year"] > max(years)]
    future_co2 = future_co2[future_co2["year"] <= 2100]
    co2_da = xr.DataArray(np.array(future_co2['value']), dims=["co2"], name="co2")
    yhat = beta * co2_da + intercept
    yhat = yhat.rename({"co2": "time"}).assign_coords(time=("time", og_time))
    yhat.to_netcdf(DATA_DIR + "yhat/" + root + "_yhat.nc", mode='w')

    # Error metrics for quailty checks, these are going to tbe the most standardized versions.
    # TODO this following chunk is part of the QAQC and will probably be removed from the formal pipeline
    # TODO might want to consider masking out the 0s so that we don't run into issues with NAs, although
    # that might be overkill...
    # These are all our "insample" error metrics
    var_obs = ssp585_data.var(dim="time")
    sd_obs = var_obs ** 0.5
    sd_sim = yhat.var(dim="time") ** 0.5
    mean_obs = ssp585_data.mean(dim="time")
    mean_sim = yhat.mean(dim="time")

    resid = yhat - ssp585_data
    mse = (resid * resid).mean(dim="time")
    ns = mse / var_obs
    ct_bias = abs(mean_obs - mean_sim) / sd_obs
    ct_var = sd_sim / sd_obs

    resid.to_netcdf(DATA_DIR + "error_metrics/" + root + "_resid.nc", mode='w')
    mse.to_netcdf(DATA_DIR + "error_metrics/" + root + "_mse.nc", mode='w')
    ns.to_netcdf(DATA_DIR + "error_metrics/" + root + "_ns.nc", mode='w')
    ct_bias.to_netcdf(DATA_DIR + "error_metrics/" + root + "_ctbias.nc", mode='w')
    ct_var.to_netcdf(DATA_DIR + "error_metrics/" + root + "_ctvar.nc", mode='w')
