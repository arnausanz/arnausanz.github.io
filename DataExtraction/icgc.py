from idlelib.pyparse import trans

import rasterio.plot
# from pyproj import Transformer
import utils

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

data_name = utils.get_root_dir() + "/data/raw/icgc/cobertes-sol-2022.tif"
tiff = rasterio.open(data_name)
print(tiff.crs)

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

# TODO --> Aplicar bé aquesta funció
with rasterio.open(data_name) as dataset:
    # Defineix la coordenada (longitud, latitud) on vols consultar el valor
    coordenada = [(convert_lat_lon_to_utm(2.154007, 41.390205))]  # Les coordenades es passen com una llista de tuples

    # Usa `sample` per obtenir el valor en aquesta coordenada
    valor_sol = list(dataset.sample(coordenada))[0][0]

    print(f"El tipus de sòl a la coordenada {coordenada[0]} és {SOIL_TYPE[valor_sol]}")
