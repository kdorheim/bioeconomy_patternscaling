# Find the data files we would like to process
import intake
import fxns

# Fetch the data from the pangeo archive - note that this might take a moment
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
full_catalog = intake.open_esm_datastore(url).df

# First limit to the NPP & RH variables, since those are going to be limiting
# we can pretty safely assume that there is going to be a corresponding tas
# result.
CARBON_VARS = ['npp', 'rh']
EXPERIMENTS = ['esm-hist', 'esm-ssp585', 'historical', 'ssp126', 'ssp585']

carbon_data = full_catalog[full_catalog["variable_id"].isin(CARBON_VARS)]
carbon_data = carbon_data[carbon_data["experiment_id"].isin(EXPERIMENTS)]
carbon_data = carbon_data[carbon_data["table_id"] == "Lmon"]

# Let's make sure that there is full ensemble & variable coverage.
carbon_meta_data = carbon_data[['activity_id', 'institution_id', 'source_id',
                                'experiment_id', 'member_id', 'variable_id', 'grid_label']]
carbon_meta_data.drop_duplicates()
carbon_meta_data['exists'] = 1

# Make sure that we have a data file coverage we need for the carbon variables.
carbon_coverage = carbon_meta_data.pivot(index=['institution_id', 'source_id',
                                                'experiment_id', 'member_id', 'grid_label'],
                                         columns='variable_id', values='exists').reset_index().dropna()
full_coverage = carbon_coverage[["source_id", "member_id", "grid_label",  "experiment_id"]].drop_duplicates()
# Subset the carbon data of interest for the files that we know have full coverage.
carbon_to_process = fxns.subset_join(full_coverage, carbon_data)


# Find the area data files - these are needed for processing.
AREA_VARS = ['areacella', 'sftlf']
area_files = full_catalog[full_catalog["variable_id"].isin(AREA_VARS)]
area_files = area_files.pivot(index=['activity_id', 'institution_id', 'source_id',
                                     'experiment_id', 'member_id', 'grid_label'],
                              columns='variable_id', values='zstore').reset_index().dropna()
carbon_w_area = fxns.subset_join(carbon_to_process, area_files)

# Make sure that there is full coverage between the scenarios & ensemble members.
exp_ens = carbon_w_area[['experiment_id', 'member_id']].drop_duplicates()
exp_ens['exists'] = 1
complete_ens = exp_ens.pivot(index="member_id", columns='experiment_id').dropna().reset_index()['member_id']

# Subset the carbon area so that it only contains the complete ensemble members.
to_process = carbon_w_area[carbon_w_area["member_id"].isin(complete_ens)]

# TODO dev related note!
# This still ends up being lots files to process! During the dev stage
# let's narrow this down further
to_process = to_process[to_process['source_id'] == "CanESM5"]
# For now let's keep using these two ensemble members since we used them in the past
to_process = to_process[to_process['member_id'].isin(["r10i1p1f1", 'r4i1p1f1'])]
to_process.to_csv("npp_rh_to_process.csv")


