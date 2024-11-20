import pandas as pd
import rasterio.plot
from rasterio.windows import Window
from pyproj import Transformer
import utils
from tqdm import tqdm
import math

"""
This file contains the code to extract the soil cover data from the ICGC dataset.
It reads the TIFF file containing the soil cover data, samples a low percentage of the pixels, save a CSV file with it's information.
"""

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
output_file = utils.get_root_dir() + '/data/processed/icgc/cobertes_sol.csv'
output_crs = 'EPSG:4326'


# Open the TIFF image
with rasterio.open(input_file) as dataset:
    # Get the transform information
    transform_affine = dataset.transform
    input_crs = dataset.crs
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

    pixel_classes = []
    # Calculate a low percentage of the total pixels to sample
    total_pixels = dataset.width * dataset.height
    target_sample_size = math.ceil(total_pixels * 0.000025)  # 0.0025% of the total pixels
    stride = int(math.sqrt(total_pixels / target_sample_size))
    # Print total pixels, target sample size, and stride
    print(f"Total Pixels: {total_pixels}, Target Sample Size: {target_sample_size}, Stride: {stride}")

    # Use a progress bar to track the processing
    num_rows = dataset.height // stride
    num_cols = dataset.width // stride
    total_samples = num_rows * num_cols
    # Use the progress bar
    with tqdm(total=total_samples, desc="Processing sampled pixels") as pbar:
        for i in range(0, dataset.height, stride):
            for j in range(0, dataset.width, stride):
                # Read only a single pixel at (i, j)
                window = Window(j, i, 1, 1)
                data = dataset.read(1, window=window)
                transform_affine_chunk = dataset.window_transform(window)
                # Get the pixel class
                class_value = data[0, 0]
                # Convert pixel coordinates to latitude and longitude
                x_src, y_src = rasterio.transform.xy(transform_affine_chunk, 0, 0, offset='center')
                x_lon, y_lat = transformer.transform(x_src, y_src)
                # Append latitude, longitude, and class to the list
                pixel_classes.append((y_lat, x_lon, class_value))
                # Update the progress bar
                pbar.update(1)

# Create a DataFrame from the pixel_classes list and save it to a CSV file
df = pd.DataFrame(pixel_classes, columns=['Latitude', 'Longitude', 'Class'])
df.to_csv(output_file, index=False)
