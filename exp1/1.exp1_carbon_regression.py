# Let's try predicting the future Rh and NPP using the historical weights, this will let us
# get a sense of if we can use the historical weights from the TRENDY results.

# 0. Set Up ------------------------------------------------------------------------------------------------------------
import os
import xarray as xr

# Prep where to write the processed data out to
OUT_DIR = "./exp1/data/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR+"error_metrics", exist_ok=True)

def get_yhat(beta, intercept, global_ts):

    VAR = list(global_ts.data_vars)
    yhat = beta.field_polyfit_coefficients * global_ts[VAR]+ intercept.field_polyfit_coefficients
    yhat = yhat.transpose(*("lat", "lon", "time"))
    # debugging
    #yhat.min(skipna=True)
    #yhat.max(skipna=True)
    #yhat.mean(skipna=True)

    return yhat

def get_errs(yhat, field):
    var_obs = field.var(dim="time")
    sd_obs = var_obs ** 0.5
    sd_sim = yhat.var(dim="time") ** 0.5
    mean_obs = field.mean(dim="time")
    mean_sim = yhat.mean(dim="time")

    resid = yhat - field
    mse = (resid * resid).mean(dim="time")
    ns = mse / var_obs
    ct_bias = abs(mean_obs - mean_sim) / sd_obs
    ct_var = sd_sim / sd_obs

    return resid, mse, ns, ct_bias, ct_var


# 1. Main Chunk --------------------------------------------------------------------------------------------------------

# Find the global files that were generated as part of exp0
csv_files = [f for f in os.listdir("./exp0/data/global") if f.endswith(".nc")]
csv_files = [s for s in csv_files if "ssp585" in s]

for f in csv_files:
    print(f)

    # Extract the variable name and the base file name of the global data.
    before, sep, after = f.partition("Lmon_")
    VAR, sep, after2 = after.partition("_gn")
    BASE, sep, after3 = f.partition("_global")
    before, sep, after = f.partition("ssp585_")
    ENS, sep, after = after.partition("_Lmon")

    # Load the global results, these are the results that will be driving the scenarios & then also
    # the original data that will be used in the data comparisons.
    global_ds = xr.open_dataset("./exp0/data/global/" + f)
    obs_ds = xr.open_dataset("./exp0/data/field/" + BASE + "_field.nc")

    # Load the slope & intercept files, they should be from the historical scenarios for the
    # variable of interest...
    beta_files = [f for f in os.listdir("./exp0/data/beta") if f.endswith(".nc")]
    beta_files = [s for s in beta_files if "historical" in s]
    beta_files = [s for s in beta_files if VAR in s]

    intercept_files = [f for f in os.listdir("./exp0/data/intercept") if f.endswith(".nc")]
    intercept_files = [s for s in intercept_files if "historical" in s]
    intercept_files = [s for s in intercept_files if VAR in s]

    # Load the inputs!
    beta1 = xr.open_dataset("./exp0/data/beta/" + beta_files[0])
    beta2 = xr.open_dataset("./exp0/data/beta/" + beta_files[1])

    intercept1 = xr.open_dataset("./exp0/data/intercept/" + intercept_files[0])
    intercept2 = xr.open_dataset("./exp0/data/intercept/" + intercept_files[1])

    # Generate the y hat values
    yhat1 = get_yhat(beta1, intercept1, global_ds)
    yhat2 = get_yhat(beta2, intercept2, global_ds)

    # Calculate the error metrics
    resid1, mse1, ns1, ct_bias1, ct_var1 = get_errs(yhat1, obs_ds)
    resid2, mse2, ns2, ct_bias2, ct_var2 = get_errs(yhat2, obs_ds)


    yhat1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_yhat1.nc", mode="w")
    yhat2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_yhat2.nc", mode="w")

    resid1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_resid1.nc", mode="w")
    resid2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_resid2.nc", mode="w")

    mse1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_mse1.nc", mode="w")
    mse2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_mse2.nc", mode="w")

    ns1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ns1.nc", mode="w")
    ns2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ns2.nc", mode="w")

    ct_bias1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ctbias1.nc", mode="w")
    ct_bias2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ctbias2.nc", mode="w")

    ct_var1.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ctvar1.nc", mode="w")
    ct_var2.to_netcdf(OUT_DIR+"error_metrics/" + VAR + "_" + ENS + "_ctvar2.nc", mode="w")