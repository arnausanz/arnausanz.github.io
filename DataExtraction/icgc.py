import rasterio
import rasterio.plot
import utils

data_name = utils.get_root_dir() + "/data/raw/icgc/cobertes-sol-2022.tif"
tiff = rasterio.open(data_name)