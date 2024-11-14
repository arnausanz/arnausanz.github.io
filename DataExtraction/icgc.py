import pandas as pd
import rasterio.plot
# from pyproj import Transformer
import utils
from rasterio.warp import transform

# Dictionary with the different types of soil
SOIL_TYPE = {
    1: "Conreus herbacis",
    2: "Horta, vivers i conreus forçats",
    3: "Vinyes",
    4: "Oliverars",
    5: "Altres conreus llenyosos",
    6: "Conreus en transformació",
    7: "Boscos densos d'aciculiòfils",
    8: "Boscos densos de caducifolis, planifolis",
    9: "Boscos densos d'esclerofil·les i laurifolis",
    10: "Matollar",
    11: "Boscos clars d'aciculiòfils",
    12: "Boscos clars de caducifolis, planifolis",
    13: "Boscos clars d'esclerofil·les i laurifolis",
    14: "Prats i herbassars",
    15: "Bosc de ribera",
    16: "Sòl no forestal",
    17: "Zones cremades",
    18: "Roquissars i congestes",
    19: "Platges",
    20: "Zones humides",
    21: "Casc urbà",
    22: "Eixample",
    23: "Zones Urbanes laxes",
    24: "Edificacions aïllades en l'espai rural",
    25: "Àrees residencials aïllades",
    26: "Zones verdes",
    27: "Zones industrials, comercials i/o de serveis",
    28: "Zones esportives i de lleure",
    29: "Zones d'extracció minera i/o abocadors",
    30: "Zones en transformació",
    31: "Xarxa viària",
    32: "Sòl nu urbà",
    33: "Zones aeroportuàries",
    34: "Xarxa ferroviària",
    35: "Zones portuàries",
    36: "Embassaments",
    37: "Llacs i llacunes",
    38: "Cursos d'aigua",
    39: "Basses",
    40: "Canals artificials",
    41: "Mar",
    0: "Sense dades"
}

input_file = utils.get_root_dir() + '/data/raw/icgc/cobertes-sol-2022.tif'
output_file = utils.get_root_dir() + '/data/processed/icgc/cobertes-sol.csv'

# Open the TIFF image
with rasterio.open(input_file) as dataset:
    # Read the image data
    data = dataset.read(1)  # Read the first band (assuming single-band image)

    # Prepare to store results
    pixel_classes = []

    # Get the transform information (affine transformation for pixel-to-coordinate mapping)
    transform_affine = dataset.transform
    src_crs = dataset.crs  # Get the original CRS of the dataset

    # Define the target CRS for latitude and longitude (WGS84)
    dst_crs = 'EPSG:4326'

    # Loop over each pixel in the image
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            # Get the pixel value (class) at this location
            class_value = data[row, col]

            # Convert pixel coordinates to the source spatial coordinates
            x_src, y_src = rasterio.transform.xy(transform_affine, row, col, offset='center')

            # Reproject coordinates to latitude and longitude if necessary
            if src_crs != dst_crs:
                x_lon, y_lat = transform(src_crs, dst_crs, [x_src], [y_src])
                x_lon, y_lat = x_lon[0], y_lat[0]  # Unpack from lists
            else:
                x_lon, y_lat = x_src, y_src
            #print("latitude ", x_lon, " longitude ", y_lat)
            # Append the latitude, longitude, and class to the list
            #pixel_classes.append((x_lon, y_lat, class_value))
            pixel_classes.append((y_lat, x_lon, class_value))
        if row == 10:
            break

# Example: print first 10 pixels
print(pixel_classes[:10])
df = pd.DataFrame(pixel_classes, columns=['Latitude', 'Longitude', 'Class'])
df.to_csv(output_file, index=False)

# Define the transformer to convert from latitude and longitude to UTM coordinates
# transformer = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)

def convert_lat_lon_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates
    :param lat: Latitude
    :param lon: Longitude
    :return: UTM coordinates
    """
    # x, y = transformer.transform(lat, lon)
    # return x, y
    pass

print(convert_lat_lon_to_utm(2.154007, 41.390205))
