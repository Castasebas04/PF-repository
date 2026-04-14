import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Definir el pipeline de transformaciones
transformaciones = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=200, sigma=15, p=1),
    A.GridDistortion(num_steps=5, distort_limit=0.5, p=1),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1),
])

# 2. Rutas de tus archivos
ruta_imagen = r'C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\tile og.png'
ruta_mascara = r'C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\mascara.png'

# 3. Lectura segura 
# Leemos la imagen a color
imagen_array = np.fromfile(ruta_imagen, np.uint8)
imagen = cv2.imdecode(imagen_array, cv2.IMREAD_COLOR)
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Leemos la máscara en escala de grises
mascara_array = np.fromfile(ruta_mascara, np.uint8)
mascara = cv2.imdecode(mascara_array, cv2.IMREAD_GRAYSCALE)

# 4. Aplicar la transformación simultánea
datos_aumentados = transformaciones(image=imagen, mask=mascara)
imagen_aumentada = datos_aumentados['image']
mascara_aumentada = datos_aumentados['mask']

# 5. Visualizar
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(imagen)
ax[0, 0].set_title('Imagen Original')
ax[0, 1].imshow(mascara, cmap='gray')
ax[0, 1].set_title('Máscara Original')

ax[1, 0].imshow(imagen_aumentada)
ax[1, 0].set_title('Imagen Aumentada')
ax[1, 1].imshow(mascara_aumentada, cmap='gray')
ax[1, 1].set_title('Máscara Aumentada (Alineada)')

for a in ax.flat:
    a.axis('off')

plt.tight_layout()
plt.show()
