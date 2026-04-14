import numpy as np
import cv2
import os

def leer_imagen(ruta):
    """Lee imágenes en rutas con tildes o caracteres especiales en Windows"""
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def guardar_imagen(ruta, imagen):
    """Guarda imágenes en rutas con tildes o caracteres especiales en Windows"""
    cv2.imencode('.png', imagen)[1].tofile(ruta)

def get_mean_and_std(x):
    """Calcula la media y desviación estándar y las aplana."""
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std

if __name__ == "__main__":
    # 1. Configura rutas
    input_dir = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\IMG BR-007-PAS-25"
    output_dir = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\Prueba Reinhard Informe"
    template_path = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\ref hye.png"

    # Crear carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Leer y calcular estadísticas de la imagen plantilla (Template)
    print("Cargando imagen template...")
    
  
    template_img = leer_imagen(template_path)
    
    if template_img is None:
        print(f"ERROR: No se pudo cargar el template. Revisa la ruta:\n{template_path}")
        exit()

    # Convertir a LAB y a float32 para la matemática
    template_lab = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB).astype("float32")
    t_mean, t_std = get_mean_and_std(template_lab)

    # Obtener lista de imágenes a procesar
    input_image_list = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"Iniciando normalización de {len(input_image_list)} imágenes...")

    # 3. Bucle para procesar cada imagen
    for img_name in input_image_list:
        source_path = os.path.join(input_dir, img_name)
        
        # --- Leer con la función especial ---
        input_img = leer_imagen(source_path)
        
        if input_img is None:
            continue
            
        source_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB).astype("float32")
        
        # Estadísticas de esta imagen específica
        s_mean, s_std = get_mean_and_std(source_lab)
        
        # Separar canales L, A, B
        l, a, b = cv2.split(source_lab)
        
        # Matemática de Reinhard Vectorizada
        l = ((l - s_mean[0]) * (t_std[0] / (s_std[0] + 1e-5))) + t_mean[0]
        a = ((a - s_mean[1]) * (t_std[1] / (s_std[1] + 1e-5))) + t_mean[1]
        b = ((b - s_mean[2]) * (t_std[2] / (s_std[2] + 1e-5))) + t_mean[2]
        
        # Limitar valores a 0-255
        l = np.clip(l, 0, 255)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        
        # Volver a unir y convertir a color normal (BGR)
        transfer = cv2.merge([l, a, b]).astype("uint8")
        transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
        
        # --- Guardar con la función especial ---
        output_path = os.path.join(output_dir, "modified_" + img_name)
        guardar_imagen(output_path, transfer)
        
        # Imprime el progreso en la misma línea
        print(f"Normalizado: {img_name}", end='\r')

    print("\n✅ Proceso completado exitosamente.")
