# API de Reconocimiento Facial Gycso con Python y DeepFace

Una API moderna de reconocimiento facial construida con FastAPI y DeepFace que permite comparar rostros utilizando imágenes codificadas en base64 y almacenamiento en AWS S3.

## Características

- Comparar dos imágenes codificadas en base64 para determinar si contienen la misma persona
- Comparar una imagen codificada en base64 con una imagen almacenada en AWS S3
- Procesamiento de imágenes en memoria (sin archivos temporales almacenados en disco)
- Modelos de reconocimiento facial configurables
- Documentación interactiva de la API con Swagger UI

## Instalación y Configuración Local

### Prerequisitos

- Python 3.8 o superior
- pip (instalador de paquetes de Python)
- Git

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/yourusername/face-recognition-api.git
cd face-recognition-api
```

### Paso 2: Crear y Activar un Entorno Virtual (Opcional pero Recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará todos los paquetes requeridos incluyendo:
- deepface
- fastapi
- python-multipart
- uvicorn
- boto3
- pillow
- numpy
- opencv-python

### Paso 4: Configurar AWS S3 (Opcional - Solo si se Utiliza la Integración con S3)

Configurar variables de entorno para AWS S3:

```bash
# En Windows
set AWS_ACCESS_KEY_ID=your_access_key
set AWS_SECRET_ACCESS_KEY=your_secret_key
set AWS_REGION=your_region
set S3_BUCKET=your_bucket_name

# En macOS/Linux
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=your_region
export S3_BUCKET=your_bucket_name
```

### Paso 5: Ejecutar el Servidor API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en http://localhost:8000

## Uso de la API

### Documentación Interactiva de la API

La API viene con documentación interactiva integrada mediante Swagger UI. Para acceder:

1. Inicie el servidor API como se describió anteriormente
2. Abra su navegador y vaya a http://localhost:8000/docs

En la interfaz de Swagger UI, puede:
- Ver todos los endpoints disponibles
- Expandir cada endpoint para ver los parámetros de solicitud
- Probar la API directamente completando los formularios de solicitud
- Ver esquemas de respuesta y códigos de estado

Alternativamente, puede acceder a la documentación ReDoc en http://localhost:8000/redoc para un diseño de documentación diferente.

### Endpoints Disponibles

1. **POST /compare-base64**: Comparar dos imágenes codificadas en base64
2. **POST /compare-with-s3**: Comparar una imagen codificada en base64 con una imagen almacenada en S3

## Modelos de Reconocimiento Facial

DeepFace soporta múltiples modelos de reconocimiento facial de última generación. Puede especificar qué modelo usar estableciendo el parámetro `model_name` en la función DeepFace.verify.

| Modelo | Descripción | Tamaño | Precisión | Velocidad | Notas |
|-------|-------------|------|----------|-------|-------|
| VGG-Face | Basado en la arquitectura VGG-16 | 515 MB | 96.7% | Media | Modelo predeterminado, buen equilibrio entre precisión y velocidad |
| Facenet | Modelo FaceNet de Google | 91 MB | 97.4% | Rápida | Bueno para dispositivos móviles y de borde |
| Facenet512 | FaceNet mejorado con embeddings de 512D | 92 MB | 98.4% | Rápida | Mayor precisión que el FaceNet original |
| OpenFace | Reconocimiento facial de código abierto | 93 MB | 78.7% | Rápida | Ligero pero menos preciso |
| DeepFace | Modelo DeepFace de Facebook | 95 MB | 69.0% | Media | Importancia histórica pero superado por modelos más nuevos |
| DeepID | Características de identidad oculta profunda | 29 MB | 66.5% | Rápida | Compacto pero menos preciso |
| ArcFace | Pérdida de margen angular aditivo | 112 MB | 96.7% | Media | Excelente para ángulos y iluminación desafiantes |
| Dlib | Modelo ResNet de Dlib | 94 MB | 96.8% | Lenta | Robusto pero más lento que otras opciones |
| SFace | Modelo eficiente de reconocimiento facial | 112 MB | 93.0% | Rápida | Bueno para entornos con recursos limitados |
| GhostFaceNet | Reconocimiento facial ligero | 24 MB | 93.3% | Muy Rápida | Mejor para aplicaciones móviles |

### Comparación de Rendimiento de Modelos

La siguiente tabla muestra las puntuaciones de precisión medidas en DeepFace en comparación con las puntuaciones reportadas en los estudios originales:

| Modelo | Puntuación Medida | Puntuación Declarada |
|-------|---------------|----------------|
| Facenet512 | 98.4% | 99.6% |
| Seres humanos | 97.5% | 97.5% |
| Facenet | 97.4% | 99.2% |
| Dlib | 96.8% | 99.3% |
| VGG-Face | 96.7% | 98.9% |
| ArcFace | 96.7% | 99.5% |
| GhostFaceNet | 93.3% | 99.7% |
| SFace | 93.0% | 99.5% |
| OpenFace | 78.7% | 92.9% |
| DeepFace | 69.0% | 97.3% |
| DeepID | 66.5% | 97.4% |

### Elegir el Modelo Adecuado

- Para mayor precisión: **Facenet512**
- Para mejor equilibrio velocidad/precisión: **Facenet** o **VGG-Face**
- Para entornos con recursos limitados: **GhostFaceNet**
- Para poses e iluminación desafiantes: **ArcFace**

## Backends de Detección Facial

DeepFace soporta múltiples backends de detección facial:

| Detector | Velocidad | Precisión | Notas |
|----------|-------|----------|-------|
| opencv | Muy Rápida | Moderada | Predeterminada, mejor para velocidad |
| ssd | Rápida | Buena | Buen equilibrio entre velocidad y precisión |
| dlib | Media | Buena | Detección confiable |
| mtcnn | Lenta | Muy Buena | Precisa pero más lenta |
| retinaface | Lenta | Excelente | Mejor precisión, especialmente para imágenes desafiantes |
| mediapipe | Rápida | Buena | Bueno para transmisiones de video |
| yolov8 | Media | Muy Buena | Bueno para múltiples rostros |
| yunet | Rápida | Buena | Detección eficiente |
| centerface | Rápida | Buena | Rápida con buena precisión |

## Métricas de Similitud

DeepFace soporta diferentes métricas de distancia para determinar la similitud entre los embeddings faciales. La elección de la métrica puede afectar tanto la precisión como la interpretación de los resultados.

| Métrica | Descripción | Umbral | Interpretación | Mejor Para |
|--------|-------------|-----------|----------------|----------|
| cosine | Mide el coseno del ángulo entre vectores | 0.68 | Menor es más similar (rango 0-1) | Predeterminado y funciona bien en la mayoría de los casos |
| euclidean | Mide la distancia en línea recta entre vectores | 100 | Menor es más similar (sin límite) | Cuando importan las magnitudes de los vectores |
| euclidean_l2 | Distancia euclidiana normalizada | 0.80 | Menor es más similar (rango 0-1) | Cuando necesitas resultados normalizados |

### Elegir la Métrica de Similitud Adecuada

La elección de la métrica de similitud depende de su caso de uso específico:

- **cosine**: Mejor para tareas generales de reconocimiento facial. Se centra en la dirección de los vectores en lugar de la magnitud, haciéndola robusta a ciertas variaciones.

- **euclidean**: Considera tanto la dirección como la magnitud. Puede ser más sensible a la escala de las características.

- **euclidean_l2**: Proporciona resultados normalizados, que pueden ser más fáciles de interpretar y establecer umbrales.

Puede especificar la métrica de similitud en sus llamadas a la API:

```python
result = DeepFace.verify(
    img1_path=img1_array,
    img2_path=img2_array,
    model_name="ArcFace",
    distance_metric="cosine"  # Opciones: "cosine", "euclidean", "euclidean_l2"
)
```

### Impacto en los Resultados de Verificación

Diferentes métricas producirán diferentes valores de distancia y utilizarán diferentes umbrales:

- Con **cosine**, dos rostros se consideran de la misma persona si la distancia es menor que 0.68
- Con **euclidean**, el umbral es típicamente alrededor de 100
- Con **euclidean_l2**, el umbral es aproximadamente 0.80

La API devuelve tanto la distancia calculada como el umbral utilizado, permitiéndole entender cuán confiado está el sistema en su resultado de verificación.

## Opciones de Configuración

La API puede configurarse con varias opciones:

- **Modelo de Reconocimiento Facial**: Cambiar el modelo utilizado para la comparación facial
- **Backend de Detección Facial**: Seleccionar el método utilizado para detectar rostros en imágenes
- **Métricas de Similitud**: Elegir entre cosine, euclidean o euclidean_l2
- **Forzar Detección**: Establecer como true para asegurar que los rostros sean detectados antes de la comparación

## Solución de Problemas

### Problemas Comunes

1. **Dependencias Faltantes**: Si encuentra errores sobre módulos faltantes, asegúrese de que todas las dependencias estén instaladas:
   ```bash
   pip install -r requirements.txt
   ```

2. **Problemas de Descarga de Modelos**: La primera vez que ejecute una comparación, DeepFace descargará los archivos de modelo requeridos. Asegúrese de tener conectividad a internet.

3. **Problemas de Memoria**: El reconocimiento facial puede requerir mucha memoria. Si encuentra errores de memoria, intente:
   - Usar un modelo más ligero como GhostFaceNet
   - Procesar imágenes más pequeñas
   - Aumentar el espacio de intercambio de su sistema

4. **Problemas de Conexión S3**: Si está utilizando la integración con S3, verifique sus credenciales de AWS y la conectividad de red.

## Agradecimientos

- [DeepFace](https://github.com/serengil/deepface) por el framework de reconocimiento facial
- [FastAPI](https://fastapi.tiangolo.com/) por el framework de API

## Verificación Facial

Esta función verifica si dos imágenes de rostro corresponden a la misma persona. El resultado incluye un valor de `distance` (distancia entre embeddings) y un `threshold` (umbral) para tomar la decisión.

### Umbrales recomendados por modelo

| Modelo       | Umbral (threshold) |
|--------------|--------------------|
| VGG-Face     | 0.40               |
| Facenet      | 0.20               |
| Facenet512   | 0.30               |
| OpenFace     | 0.42               |
| DeepFace     | 0.23               |
| DeepID       | 0.10               |
| ArcFace      | 0.40               |
| Dlib         | 0.62               |
| SFace        | 0.45               |

La comparación se considera positiva (`verified: true`) cuando:
```
distance ≤ threshold
```

### Interpretación de distancias y umbrales

- **Distance**: Representa la "distancia" o diferencia entre los dos vectores de características (embeddings). Cuanto menor sea este valor, más parecidos son los rostros.
  - Con dos imágenes idénticas, la distancia será cercana a cero (ej. 6.66e-16)
  - Con dos imágenes de la misma persona pero diferentes, la distancia será típicamente entre 0.1 y 0.3
  - Con imágenes de personas diferentes, la distancia suele ser mayor que el umbral

- **Threshold**: Es el valor límite para decidir si dos rostros pertenecen a la misma persona:
  - Si distance ≤ threshold → `verified: true` (misma persona)
  - Si distance > threshold → `verified: false` (personas diferentes)

### Ejemplos de valores de distancia:

1. **Imágenes idénticas**: distance ≈ 0
2. **Misma persona, diferente imagen**: distance ≈ 0.1-0.3
3. **Personas diferentes**: distance > umbral del modelo

Es importante notar que el valor concreto de distance dependerá del modelo utilizado y puede verse afectado por cambios en la apariencia como barba, peinado, envejecimiento, iluminación o ángulo de la foto.

### Ajuste de umbrales

- Para **mayor seguridad** (reducir falsos positivos): Utilice un umbral más bajo
- Para **mayor flexibilidad** (reducir falsos negativos): Utilice un umbral más alto

Para aplicaciones de alta seguridad, es recomendable establecer umbrales más estrictos que los valores predeterminados. 