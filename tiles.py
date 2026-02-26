import os
from PIL import Image
import numpy as np
from pathlib import Path

# Aumentar límite de píxeles para imágenes médicas gigantes
Image.MAX_IMAGE_PIXELS = None 

def cut_tiles_fixed(image_path, output_dir, tile_size=512, overlap=0, bg_threshold=0.8):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando imagen: {image_path}")
    img = Image.open(image_path)
    width, height = img.size
    print(f"Dimensiones: {width}x{height}")
    
    stride = tile_size - overlap
    saved_count = 0
    
    # CAMBIO 1: El rango va hasta el final (width y height completos)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            
            # CAMBIO 2: Definir coordenadas reales (sin salirse de la imagen)
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # Extraer el recorte (puede ser más pequeño que 512x512 en los bordes)
            tile = img.crop((x, y, x_end, y_end))
            
            # CAMBIO 3: Padding (Relleno) si el tile es pequeño
            # Si el tile no es de 512x512, creamos un lienzo blanco y pegamos el tile
            if tile.size != (tile_size, tile_size):
                new_tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255)) # Fondo blanco
                new_tile.paste(tile, (0, 0))
                tile = new_tile
            
            # Verificar fondo (Tu lógica original)
            if not is_mostly_background(tile, bg_threshold):
                tile_name = f"tile_x{x:05d}_y{y:05d}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile.save(tile_path)
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"Guardados {saved_count} tiles...", end='\r')

    print(f"\n✅ Proceso completado. Tiles guardados: {saved_count}")

def is_mostly_background(tile, threshold=0.8):
    # Optimización: Convertir a 'L' (Grayscale) directamente con PIL es más rápido que Numpy
    gray = np.array(tile.convert('L'))
    
    # 210 es un buen umbral para "beige/blanco"
    white_pixels = np.sum(gray > 140) 
    white_ratio = white_pixels / gray.size
    
    return white_ratio > threshold

if __name__ == "__main__":
    # Recuerda usar r"" para las rutas en Windows para evitar errores con \
    image_path = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\Imagenes\BR-007-TRICROMICO-25.tif"
    output_dir = r"C:\Users\Sebastián\Documents\Uninorte\9no semestre\PF\PF repository\tiles_img"
    
    cut_tiles_fixed(image_path, output_dir)