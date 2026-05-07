# This R script is used to extract shaping information that can be used 
# to map the nc coordinates to the GCAM land use regions. 

# Warning it took quite a while to download and install the gaia packagge
# remotes::install_github("jgcri/gaia") 
library(dplyr)


lazyLoad(
  file.path(system.file("R", package='gaia'), 'sysdata')
) 

write.csv(mapping_rmap_grid, 'GCAM_grid_region_basin.csv', row.names=FALSE)
