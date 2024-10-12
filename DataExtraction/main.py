import os

# Obtener la ruta absoluta al directorio raíz del repositorio
repo_root = os.getcwd()

# Crear una ruta absoluta para el archivo que deseas escribir
file_path = os.path.join(repo_root, 'Data', 'test.txt')

with open(file_path, "a") as file:
    """
    Testing automation --> Write the day and time of the execution in a file
    """
    file.write("Day: " + str(os.popen("date +'%d/%m/%Y'").read()).strip() + ' - ' + "Time: " + str(os.popen("date +'%T'").read()).strip())
    file.write("\n")
    file.close()