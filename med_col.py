from PIL import Image
import numpy as np

# Pega aquí la ruta de UNO de los tiles vacíos que se colaron
ruta_tile_vacio = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\tiles_img\tile_x01536_y46080.png"
img = Image.open(ruta_tile_vacio).convert('L') # Convertir a escala de grises
array = np.array(img)

promedio = np.mean(array)
minimo = np.min(array)
maximo = np.max(array)

print(f"--- Análisis del Fondo ---")
print(f"Brillo Promedio: {promedio:.2f} (de 255)")
print(f"Píxel más oscuro: {minimo}")
print(f"Píxel más claro: {maximo}")
print(f"--------------------------")
print(f"RECOMENDACIÓN: Pon tu threshold en {int(minimo - 10)}")