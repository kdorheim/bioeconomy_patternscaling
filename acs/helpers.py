# Helper/convenience functions for Pangeo-Enabled ESM Pattern Scaling.
# #####################################################
# Load all of the libraries
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import intake
import fsspec


# #####################################################
# Helper function to easily pull a netcdf from pangeo with just the zstore address from
# the stitches `pangeo_table.csv` file copied into this project.
def fetch_nc(zstore):
    """Extract data for a single file.
    :param zstore:                str of the location of the cmip6 data file on pangeo.
    :return:                      an xarray containing cmip6 data downloaded from the pangeo.
    """
    ds = xr.open_zarr(fsspec.get_mapper(zstore))
    ds.sortby('time')
    return ds

# #####################################################
# Helper functions to calculate Tgav from each file easily from
# https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/ECS_Gregory_method.html

def get_lat_name(ds):
    """Figure out what is the latitude coordinate for each dataset."""
    for lat_name in ['lat', 'latitude']:
        if lat_name in ds.coords:
            return lat_name
    raise RuntimeError("Couldn't find a latitude coordinate")

def global_mean(ds):
    """Return global mean of a whole dataset."""
    lat = ds[get_lat_name(ds)]
    weight = np.cos(np.deg2rad(lat))
    weight /= weight.mean()
    other_dims = set(ds.dims) - {'time'}
    return (ds * weight).mean(other_dims)

# #####################################################
# Helper function for doing regression on an xarray of monthly or annual data
# (all separated out, eg you have an xarray of only the Januaries, only the
# Februaries, only the annual values, etc
# return an xarray with the slope, intercept, and residuals

def pattern_scale(xarraydata_stacked, tgav, fit_intercept, save_resids=False):
    # set up for regression
    varname = list(xarraydata_stacked.keys())[0]


    # independent var
    x = tgav['tas'].values.reshape(-1,1) # -1 means that calculate the dimension of rows, but have 1 column

    # dependent var
    y = xarraydata_stacked[varname].values

    # create an object of the sklearn Linear Regression class
    linear_regressor = LinearRegression(fit_intercept=fit_intercept)


    # perform regression
    linear_regressor.fit(x,y)

    # get estimated parameters
    slope = linear_regressor.coef_.T
    if fit_intercept:
        intercept = linear_regressor.intercept_.reshape(slope.shape)
    # ^ regressor.coef_ is a 1d array
    # and regressor.intercept_ is a vector so we force
    # it to be a 1d array too, to make things easier


    if not fit_intercept:
        out_data = xr.Dataset({'slope': xr.DataArray(
            slope,
            coords=[['estimated_param'], xarraydata_stacked.unifiedspatial],
            dims=['param', 'unifiedspatial'])
        })

    if fit_intercept:
        out_data = xr.Dataset({'slope': xr.DataArray(
            slope,
            coords=[['estimated_param'], xarraydata_stacked.unifiedspatial],
            dims=['param', 'unifiedspatial']),
            'intercept': xr.DataArray(
                intercept,
                coords=[['estimated_param'], xarraydata_stacked.unifiedspatial],
                dims=['param', 'unifiedspatial'])
        })

        if save_resids:
            # get residuals
            resids = y - linear_regressor.predict(x)

            # add them to the original xarray by making a copy and
            # overwriting the tas values. This makes sure
            # everything is the same dimension so that we can just
            # assign this into the original xarraydata
            z = xarraydata_stacked.copy()
            z[varname].values = resids
            #z = z.rename_vars({varname: varname + '_resids'})
            out_data =  [out_data, z]


    return out_data

# #####################################################
# Helper function for reshaping an xarray of monthly  data for easier
# pattern scaling on monthly or annual data, and then to call the
# pattern_scale() function.
# For monthly patterns, from a full xarray of CMIP6-style monthly data, the
# xarray is separated into a list of 12 xarray datasets: a data set of all
# Januaries, a data set of all Februaries, etc.
# for Annual data, the xarray of  CMIP6-style monthly data is used to calculate
# annual average values in each grid cell.
#
# return an xarray with the slope, intercept, and residuals when specified by
# the fit_intercept and save_resids arguments.
def reshape_and_pattern_scale(xarraydata, tgav,  monthly_or_annual,
                              fit_intercept, save_resids=False):
    if monthly_or_annual == 'monthly':
        stacked = xarraydata.stack(unifiedspatial=['lat', 'lon']).copy()
        months = pd.DataFrame(data={
            'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'month_name': ['jan', 'feb', 'mar', 'apr',
                           'may', 'jun', 'jul', 'aug',
                           'sep', 'oct', 'nov', 'dec']
        })

        monthly_list = []
        for ind in range(1, 13):
            month_nm = months.loc[(ind - 1), 'month_name']

            tas_mon = stacked.isel(time=slice((ind - 1), None, 12)).copy()

            mon_pattern = pattern_scale(xarraydata_stacked=tas_mon,
                                        tgav=tgav,
                                        fit_intercept=fit_intercept,
                                        save_resids=save_resids)
            # inherit attributes
            if not save_resids:
                mon_pattern = mon_pattern.unstack('unifiedspatial').copy()
                mon_pattern.attrs = xarraydata.attrs
                mon_pattern.attrs['pattern_info'] = ('pattern of ' + month_nm + ' gridded data')
            if save_resids:
                mon_pattern[0] = mon_pattern[0].unstack('unifiedspatial').copy()
                mon_pattern[0].attrs = xarraydata.attrs
                mon_pattern[0].attrs['pattern_info'] = ('pattern of ' + month_nm + ' gridded data')
                mon_pattern[1] = mon_pattern[1].unstack('unifiedspatial').copy()
                mon_pattern[1].attrs = xarraydata.attrs
                mon_pattern[1].attrs['pattern_info'] = ('pattern of ' + month_nm + ' gridded data - resids')
            monthly_list.append(mon_pattern)
            #end for loop
        out_data=monthly_list
        # end if monthly patterns


    if monthly_or_annual == 'annual':
        xarraydata1 = xarraydata.resample(time='AS').mean('time').copy()
        stacked = xarraydata1.stack(unifiedspatial=['lat', 'lon']).copy()
        annual = pattern_scale(xarraydata_stacked=stacked,
                               tgav=tgav,
                               fit_intercept=fit_intercept,
                               save_resids=save_resids)
        ## inherit attributes
        if not save_resids:
            annual = annual.unstack('unifiedspatial').copy()
            annual.attrs = xarraydata.attrs
            annual.attrs['pattern_info'] = 'pattern of annual gridded data'
        if save_resids:
            annual[0] = annual[0].unstack('unifiedspatial').copy()
            annual[0].attrs = xarraydata.attrs
            annual[0].attrs['pattern_info'] = 'pattern of annual gridded data'
            annual[1] = annual[1].unstack('unifiedspatial').copy()
            annual[1].attrs = xarraydata.attrs
            annual[1].attrs['pattern_info'] = 'pattern of annual gridded data - resids'
        out_data = annual.copy()
        # end if annual patterns


    return out_data



# #####################################################
# wrapper function to take esm, experiment, variable, monthly_or_annual
# patterns, pull all ensemble members for that ESM and experiment from
# Pangeo as CMIP6 monthly netcdfs, calculate the ensemble average values,
# and then call pattern scaling.

def do_ps(esm_name, exp_name, var_name, monthly_or_annual, fit_intercept, save_resids=False, tgav_DIR = ''):

    # ###################################################
    # Step 1: get the ensemble average data
    # ###################################################
    #
    # ###################################################
    # Step 1a: prep list of netcdf addresses on pangeo
    # ###################################################
    pangeo_data = pd.read_csv('pangeo_table.csv')
    # get the relevant address
    nc_address = pangeo_data[((pangeo_data['model'] == esm_name))
                             & ((pangeo_data['variable'] == var_name))
                             & ((pangeo_data['experiment'] == exp_name))
                             & ((pangeo_data['domain'].str.contains('mon')))].drop_duplicates().copy()

    # Keep only p1 ensemble members
    nc_address = nc_address[nc_address['ensemble'].str.contains('p1')].drop_duplicates().reset_index(drop=True).copy()
    # isolate the file list for opening to calculate the ensemble average data
    file_list = np.unique(nc_address.zstore.values.flatten())

    if len(file_list) ==0:
        print('==================================================')
        print('This combination of data: ')
        print( esm_name + ' ' + var_name + ' ' + exp_name)
        print('does not exist in pangeo_data.csv. Skipping.')
        print('==================================================')
        return(None)

    #return(file_list)

    # ###################################################
    # Step 1b:   create the holder xarray for the ensemble average data
    # ###################################################
    # First, read in a single ensemble member so that we have the
    # xarray data structure
    ensemble_ds = []
    file_list_ind = 0
    # use a while loop to make sure we are only using a good, complete esm realization
    while len(ensemble_ds) == 0:
        x = fetch_nc(file_list[file_list_ind])
        v = list(x.keys())[0]
        variable_attrs = x[v].attrs.copy()
        del (v)

        # some ensemble members go to 2300,
        # cut off at 2100 to work with.
        # QC not robust to anything outside
        # of historical or ssps.
        if (exp_name != 'historical'):
            x = x.sel(time=slice('2015', '2100')).copy()
            if (len(x.time) >= 12 * 85):
                attr_copy = x.attrs.copy()
                new_attrs = dict((k, attr_copy[k]) for k in ['activity_id',
                                                             'branch_method',
                                                             'experiment',
                                                             'experiment_id',
                                                             'external_variables',
                                                             'forcing_index',
                                                             'frequency',
                                                             'further_info_url',
                                                             'grid',
                                                             'grid_label',
                                                             'initialization_index',
                                                             'mip_era',
                                                             'source_id',
                                                             'variable_id']
                                 if k in attr_copy)
                x.attrs = new_attrs
                x.attrs['variant_label'] = 'ensemble_avg'
                x.attrs['variable_units'] = variable_attrs['units']
                ensemble_ds = x.copy()

        if (exp_name == 'historical'):
            x = x.sel(time=slice('1850', '2014')).copy()
            if (len(x.time) >= 12 * 164):
                attr_copy = x.attrs.copy()
                new_attrs = dict((k, attr_copy[k]) for k in ['activity_id',
                                                             'branch_method',
                                                             'experiment',
                                                             'experiment_id',
                                                             'external_variables',
                                                             'forcing_index',
                                                             'frequency',
                                                             'further_info_url',
                                                             'grid',
                                                             'grid_label',
                                                             'initialization_index',
                                                             'mip_era',
                                                             'source_id',
                                                             'variable_id']
                                 if k in attr_copy)
                x.attrs = new_attrs
                x.attrs['variant_label'] = 'ensemble_avg'
                x.attrs['variable_units'] = variable_attrs['units']
                ensemble_ds = x.copy()

        del (x)
        del (variable_attrs)
        file_list_ind = file_list_ind + 1
        # end while loop

    # ###################################################
    # Step 1c:   Calculate the ensemble average
    # ###################################################

    df_sum = 0
    n_good_files = 0
    for zstore in file_list:
        print(zstore)
        x = fetch_nc(zstore)

        # some ensemble members go to 2300,
        # cut off at 2100 to work with.
        #QC not robust to anything outside
        # of historical or ssps
        if (exp_name != 'historical'):
            x = x.sel(time=slice('2015', '2100')).copy()
            if (len(x.time) >= 12 * 85):
                x = x[var_name].values.copy()
                df_sum = (x + df_sum).copy()
                n_good_files = n_good_files + 1

        if (exp_name == 'historical'):
            x = x.sel(time=slice('1850', '2014')).copy()
            if (len(x.time) >= 12 * 164):
                x = x[var_name].values.copy()
                df_sum = (x + df_sum).copy()
                n_good_files = n_good_files + 1

        del(x)
    # end for loop over nc_address zstore entries

    # calculate the pointwise ensemble average values
    df_ens_avg = (df_sum / n_good_files).copy()
    # replace the values in our holder xarray with the average.
    # this is definitely not the best way to do this but it does work.

    # update the values
    ensemble_ds[var_name].values = df_ens_avg.copy()

    del(df_sum)
    del(df_ens_avg)
    del(n_good_files)
    del(nc_address)
    del(file_list)

    # ###################################################
    # Step 2:   Calculate Tgav, the independent variable of our regression
    #               for any dependent variable
    # ###################################################

    if var_name == 'tas':
        ds_ann_Tgav = ensemble_ds.pipe(global_mean  # global mean in each monthXyear
                                       ).coarsen(time=12).mean()  # annual mean in each year

    # ###################################################
    # Step 2a:   if var_name is not tas, repeat all of step 1 for tas data
    #               to get tgav
    # ###################################################
    if var_name != 'tas':
        print('==================================================')
        print('non tas var being scaled, must read in tas info to get Tgav for scaling.')
        print('Doing now:.')
        print('==================================================')

        nc_addressT = pangeo_data[((pangeo_data['model'] == esm_name))
                                 & ((pangeo_data['variable'] == 'tas'))
                                 & ((pangeo_data['experiment'] == exp_name))
                                 & ((pangeo_data['domain'].str.contains('mon')))].drop_duplicates().copy()

        # Keep only p1 ensemble members
        nc_addressT = nc_addressT[nc_addressT['ensemble'].str.contains('p1')].drop_duplicates().reset_index(
            drop=True).copy()
        # isolate the file list for opening to calculate the ensemble average data
        file_listT = np.unique(nc_addressT.zstore.values.flatten())

        # ###################################################
        # Step 1b:   create the holder xarray for the ensemble average data
        # ###################################################
        # First, read in a single ensemble member so that we have the
        # xarray data structure
        ensemble_dsT = []
        file_list_indT = 0
        # use a while loop to make sure we are only using a good, complete esm realization
        while len(ensemble_dsT) == 0:
            x = fetch_nc(file_listT[file_list_indT])
            v = list(x.keys())[0]
            variable_attrs = x[v].attrs.copy()
            del (v)

            # some ensemble members go to 2300,
            # cut off at 2100 to work with.
            # QC  not robust to anything outside
            # of historical or ssps
            if (exp_name != 'historical'):
                x = x.sel(time=slice('2015', '2100')).copy()
                if (len(x.time) >= 12 * 85):
                    attr_copy = x.attrs.copy()
                    new_attrs = dict((k, attr_copy[k]) for k in ['activity_id',
                                                                 'branch_method',
                                                                 'experiment',
                                                                 'experiment_id',
                                                                 'external_variables',
                                                                 'forcing_index',
                                                                 'frequency',
                                                                 'further_info_url',
                                                                 'grid',
                                                                 'grid_label',
                                                                 'initialization_index',
                                                                 'mip_era',
                                                                 'source_id',
                                                                 'variable_id']
                                     if k in attr_copy)
                    x.attrs = new_attrs
                    x.attrs['variant_label'] = 'ensemble_avg'
                    x.attrs['variable_units'] = variable_attrs['units']
                    ensemble_dsT = x.copy()

            if (exp_name == 'historical'):
                x = x.sel(time=slice('1850', '2014')).copy()
                if (len(x.time) >= 12 * 164):
                    attr_copy = x.attrs.copy()
                    new_attrs = dict((k, attr_copy[k]) for k in ['activity_id',
                                                                 'branch_method',
                                                                 'experiment',
                                                                 'experiment_id',
                                                                 'external_variables',
                                                                 'forcing_index',
                                                                 'frequency',
                                                                 'further_info_url',
                                                                 'grid',
                                                                 'grid_label',
                                                                 'initialization_index',
                                                                 'mip_era',
                                                                 'source_id',
                                                                 'variable_id']
                                     if k in attr_copy)
                    x.attrs = new_attrs
                    x.attrs['variant_label'] = 'ensemble_avg'
                    x.attrs['variable_units'] = variable_attrs['units']
                    ensemble_dsT = x.copy()

            del (x)
            del (variable_attrs)
            file_list_indT = file_list_indT + 1
            # end while loop
        del(file_list_indT)

        # ###################################################
        # Step 1c:   Calculate the ensemble average
        # ###################################################

        df_sumT = 0
        n_good_filesT = 0
        for zstore in file_listT:
            print(zstore)
            x = fetch_nc(zstore)

            # some ensemble members go to 2300,
            # cut off at 2100 to work with.
            # QC not robust to anything outside
            # of historical or ssps
            if (exp_name != 'historical'):
                x = x.sel(time=slice('2015', '2100')).copy()
                if (len(x.time) >= 12 * 85):
                    x = x['tas'].values.copy()
                    df_sumT = (x + df_sumT).copy()
                    n_good_filesT = n_good_filesT + 1

            if (exp_name == 'historical'):
                x = x.sel(time=slice('1850', '2014')).copy()
                if (len(x.time) >= 12 * 164):
                    x = x['tas'].values.copy()
                    df_sumT = (x + df_sumT).copy()
                    n_good_filesT = n_good_filesT + 1

            del (x)
        # end for loop over nc_address zstore entries

        # calculate the pointwise ensemble average values
        df_ens_avgT = (df_sumT/ n_good_filesT).copy()
        # replace the values in our holder xarray with the average
        # this is definitely not the best way to do this but it does work.

        # update the values
        ensemble_dsT['tas'].values = df_ens_avgT.copy()

        del (df_sumT)
        del (df_ens_avgT)
        del (n_good_filesT)
        del (nc_addressT)
        del (file_listT)

        ds_ann_Tgav = ensemble_dsT.pipe(global_mean  # global mean in each monthXyear
                                       ).coarsen(time=12).mean()  # annual mean in each year
        del(ensemble_dsT)

        # end if not tas

    if len(tgav_DIR) > 0:
        ds_ann_Tgav.to_netcdf(tgav_DIR + '/' + esm_name + '_' + exp_name +'_ensemble_avg_tgav.nc')

    print('==================================================')
    print ('tgav calculation complete, doing scaling with call to reshape_and_scale.')
    print('==================================================')
    # ###################################################
    # Step 3: Do the pattern scaling
    # ###################################################
    output = reshape_and_pattern_scale(xarraydata=ensemble_ds,
                                       tgav= ds_ann_Tgav,
                                       monthly_or_annual=monthly_or_annual,
                                       fit_intercept= fit_intercept,
                                       save_resids=save_resids)

    return output




# ###########################################################


# Update the pangeo table
def fetch_pangeo_table():
    """ Get a copy of the pangeo archive contents
    :return: a pd data frame containing information about the model, source, experiment, ensemble and
    so on that is available for download on pangeo.
    """
    # # All potential experiments:
    # ['highresSST-present', 'piControl', 'hist-1950', 'control-1950',
    #    'abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2',
    #    'historical', 'esm-hist', 'esm-piControl', 'ssp245', 'ssp585',
    #    'hist-1950HC', 'piClim-2xVOC', 'piClim-2xss', 'piClim-2xdust',
    #    'piClim-histall', 'hist-piNTCF', 'piClim-lu', 'histSST',
    #    'histSST-piO3', 'histSST-piNTCF', 'piClim-2xNOx', 'piClim-CH4',
    #    'piClim-HC', 'faf-heat-NA0pct', 'piClim-O3', 'piClim-NTCF',
    #    'piClim-NOx', 'piClim-VOC', 'piClim-aer', 'hist-aer',
    #    'piClim-control', 'faf-heat', 'faf-heat-NA50pct', 'ssp370-lowNTCF',
    #    'ssp370SST-lowCH4', 'ssp370SST-lowNTCF', 'ssp370SST',
    #    'ssp370SST-ssp126Lu', 'ssp370pdSST', 'faf-all', 'piClim-anthro',
    #    'hist-nat', 'hist-GHG', 'ssp119', 'piClim-histnat', 'piClim-4xCO2',
    #    'ssp370', 'piClim-histghg', 'highresSST-future', 'piClim-ghg',
    #    'histSST-piAer', 'histSST-piCH4', 'ssp126', 'histSST-1950HC',
    #    'hist-piAer', 'faf-water', 'faf-passiveheat', 'amip', 'faf-stress',
    #    '1pctCO2-bgc', 'piClim-histaer', 'esm-ssp585-ssp126Lu', 'omip1',
    #    'esm-pi-cdr-pulse', 'esm-ssp585', '1pctCO2-rad', '1pctCO2-cdr',
    #    'esm-pi-CO2pulse', 'abrupt-solp4p', 'piControl-spinup',
    #    'hist-stratO3', 'land-hist', 'abrupt-solm4p', 'midHolocene',
    #    'lig127k', 'esm-piControl-spinup', 'ssp245-GHG', 'ssp245-nat',
    #    'dcppC-amv-neg', 'dcppC-amv-ExTrop-neg', 'dcppC-atl-control',
    #    'dcppC-amv-pos', 'dcppC-ipv-NexTrop-neg', 'dcppC-ipv-NexTrop-pos',
    #    'dcppC-atl-pacemaker', 'dcppC-amv-ExTrop-pos',
    #    'dcppC-amv-Trop-neg', 'dcppC-pac-control', 'dcppC-ipv-pos',
    #    'dcppC-pac-pacemaker', 'dcppC-ipv-neg', 'dcppC-amv-Trop-pos',
    #    'ssp460', 'ssp434', 'amip-p4K', 'amip-m4K', 'land-noLu',
    #    'hist-noLu', 'deforest-globe', 'amip-4xCO2', 'ssp534-over',
    #    'amip-future4K', 'historical-cmip5', 'hist-bgc', 'piControl-cmip5',
    #    'rcp26-cmip5', 'rcp45-cmip5', 'rcp85-cmip5', 'pdSST-piArcSIC',
    #    'pdSST-piAntSIC', 'piSST-piSIC', 'piSST-pdSIC', 'ssp245-stratO3',
    #    'amip-hist', 'hist-sol', 'hist-CO2', 'hist-volc', 'hist-totalO3',
    #    'hist-nat-cmip5', 'hist-aer-cmip5', 'hist-GHG-cmip5',
    #    'pdSST-futAntSIC', 'futSST-pdSIC', 'pdSST-pdSIC', 'ssp245-aer',
    #    'pdSST-futArcSIC', 'dcppA-hindcast', 'dcppA-assim',
    #    'dcppC-hindcast-noPinatubo', 'dcppC-hindcast-noElChichon',
    #    'dcppC-hindcast-noAgung', 'aqua-4xCO2', 'aqua-p4K', 'aqua-control',
    #    'ssp245-cov-modgreen', 'ssp245-cov-fossil', 'ssp245-cov-strgreen',
    #    'ssp245-covid', 'lgm', 'ssp585-bgc', 'piClim-SO2', 'piClim-OC',
    #    'piClim-BC', 'piClim-2xfire', 'amip-lwoff', 'amip-p4K-lwoff',
    #    '1pctCO2to4x-withism', '1pctCO2-4xext', 'hist-resIPO', 'past1000',
    #    'pa-futArcSIC', 'pa-pdSIC', 'historical-ext', 'pdSST-futArcSICSIT',
    #    'pdSST-futOkhotskSIC', 'pdSST-futBKSeasSIC', 'pa-piArcSIC',
    #    'pa-piAntSIC', 'pa-futAntSIC', 'pdSST-pdSICSIT']


    # smaller set of experiments to save to make life easier.
    # experiments most likely to want to pattern scale on or
    # otherwise get data from
    exps =  ['historical', 'ssp370',
            'ssp585', 'ssp126', 'ssp245',
            'ssp119', 'ssp460', 'ssp434',
            'ssp534-over', 'piControl']

    # The url path that contains to the pangeo archive table of contents.
    url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    dat = intake.open_esm_datastore(url)
    dat = dat.df
    out = (dat.loc[dat['grid_label'] == "gn"][["source_id", "experiment_id", "member_id", "variable_id",
                                                    "zstore", "table_id"]].copy())
    out = out.rename(columns={"source_id": "model", "experiment_id": "experiment",
                                                "member_id": "ensemble", "variable_id": "variable",
                                                "zstore": "zstore", "table_id": "domain"}).copy()
    out = (out.loc[out['experiment'].isin(exps)]).drop_duplicates().reset_index(drop=True).copy()

    return out