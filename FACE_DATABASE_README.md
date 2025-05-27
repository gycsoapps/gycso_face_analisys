# Sistema de Reconocimiento Facial con Base de Datos

## Descripción General

Este sistema extendido permite comparar rostros contra una base de datos de embeddings precomputados para control de acceso, optimizado para manejar 1000-1500 peticiones diarias comparando contra hasta 1000 rostros registrados.

## Características Principales

### 🚀 Alto Rendimiento
- **Embeddings Precomputados**: Los rostros se procesan una sola vez y se almacenan como vectores matemáticos
- **Comparación Ultra-Rápida**: Utiliza cálculo de distancia euclidiana en lugar de procesamiento de imágenes completas
- **Modelo Optimizado**: Usa Facenet por defecto (más rápido que VGG-Face)
- **Cache Inteligente**: Almacenamiento en memoria para acceso instantáneo

### 🎯 Capacidades
- Comparación de 1:N (una imagen contra toda la base de datos)
- Gestión completa de usuarios (agregar, eliminar, consultar)
- Umbrales de similitud configurables
- Respuestas detalladas con métricas de confianza
- Persistencia automática de datos

## Nuevos Endpoints

### 1. Comparación con Base de Datos
```http
POST /api/photo/compare-with-database
```

**Solicitud:**
```json
{
  "image": "base64_encoded_image",
  "user_id": "opcional_para_logs",
  "threshold": 0.6,
  "model_name": "Facenet",
  "enforce_detection": false,
  "detector_backend": "ssd"
}
```

**Respuesta:**
```json
{
  "access_granted": true,
  "match_found": true,
  "best_match": {
    "user_id": "usuario123",
    "similarity_score": 0.89,
    "distance": 0.32
  },
  "all_matches": [...],
  "processing_time": 0.15,
  "total_faces_compared": 1000
}
```

### 2. Agregar Rostro a la Base de Datos
```http
POST /api/photo/add-face-to-database
```

**Solicitud:**
```json
{
  "user_id": "usuario123",
  "image": "base64_encoded_image",
  "metadata": {"name": "Juan Pérez", "department": "IT"},
  "model_name": "Facenet",
  "enforce_detection": true,
  "detector_backend": "ssd"
}
```

### 3. Estadísticas de la Base de Datos
```http
GET /api/photo/database-stats
```

### 4. Eliminar Rostro
```http
DELETE /api/photo/remove-face/{user_id}
```

## Rendimiento Estimado

### Para tu Caso de Uso:
- **1000-1500 peticiones/día**: ✅ Soportado fácilmente
- **1000 rostros en DB**: ✅ Comparación en ~50-150ms
- **Servidor local**: ✅ Optimizado para hardware local

### Métricas de Rendimiento:
| Operación | Tiempo Estimado | Notas |
|-----------|-----------------|-------|
| Comparación 1:1000 | 50-150ms | Usando embeddings precomputados |
| Agregar nuevo rostro | 200-500ms | Incluye extracción de embedding |
| Carga inicial | 2-5 segundos | Depende del tamaño de la DB |

## Configuración Optimizada

### Variables de Entorno Recomendadas:
```env
# Modelo optimizado para velocidad/precisión
DEFAULT_MODEL=Facenet
DEFAULT_DETECTOR=ssd

# Umbral de reconocimiento (ajustable según precisión deseada)
DEFAULT_RECOGNITION_THRESHOLD=0.6

# Base de datos
EMBEDDING_DB_PATH=face_embeddings.json
PRELOAD_DATABASE_ON_STARTUP=true

# Optimizaciones
MAX_PARALLEL_COMPARISONS=10
ENABLE_EMBEDDING_CACHE=true
CACHE_SIZE_LIMIT=1000
```

## Instalación y Configuración

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Poblar Base de Datos Inicial
```bash
# Desde directorio con imágenes organizadas como: usuario1.jpg, usuario2.jpg, etc.
python populate_database.py --images_dir ./fotos_usuarios --model_name Facenet --clear_db

# Con pruebas de validación
python populate_database.py --images_dir ./fotos_usuarios --test_dir ./fotos_test --threshold 0.6
```

### 3. Ejecutar Servidor
```bash
# Desarrollo
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Producción
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Ejemplos de Uso

### Control de Acceso Básico
```python
import requests
import base64

# Leer imagen de usuario
with open("usuario_entrada.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Verificar acceso
response = requests.post("http://localhost:8000/api/photo/compare-with-database", 
    json={
        "image": image_b64,
        "threshold": 0.6
    }
)

result = response.json()
if result["access_granted"]:
    print(f"Acceso permitido para: {result['best_match']['user_id']}")
    print(f"Confianza: {result['best_match']['similarity_score']:.2%}")
else:
    print("Acceso denegado - Usuario no reconocido")
```

### Registro de Nuevo Usuario
```python
# Agregar nuevo usuario
response = requests.post("http://localhost:8000/api/photo/add-face-to-database",
    json={
        "user_id": "empleado001",
        "image": image_b64,
        "metadata": {
            "name": "Ana García",
            "department": "Recursos Humanos",
            "hired_date": "2024-01-15"
        }
    }
)
```

## Optimizaciones Adicionales Recomendadas

### Para Mayor Rendimiento:
1. **Redis Cache**: Implementar cache distribuido para embeddings
2. **FAISS Index**: Para bases de datos >10,000 rostros
3. **GPU Acceleration**: Usar TensorFlow-GPU para procesamiento más rápido
4. **Load Balancing**: Múltiples instancias del servicio

### Para Mayor Precisión:
1. **Modelos Más Grandes**: Cambiar a "VGG-Face" o "ArcFace"
2. **Múltiples Embeddings**: Almacenar varios embeddings por usuario
3. **Calidad de Imagen**: Implementar validación de calidad
4. **Entrenamiento Personalizado**: Fine-tuning con datos específicos

## Monitoreo y Métricas

### Métricas Importantes:
- Tiempo de respuesta promedio
- Tasa de falsos positivos/negativos
- Utilización de memoria
- Throughput de peticiones

### Logs:
El sistema registra automáticamente:
- Tiempo de procesamiento por petición
- Resultados de comparación
- Errores y excepciones
- Estadísticas de la base de datos

## Estructura de Archivos

```
proyecto/
├── app.py                     # Servidor principal con nuevos endpoints
├── face_service.py            # Lógica de reconocimiento + base de datos
├── models.py                  # Modelos de datos para nuevos endpoints
├── populate_database.py       # Script para poblar DB inicial
├── face_embeddings.json       # Base de datos de embeddings (auto-generado)
├── requirements.txt           # Dependencias actualizadas
└── FACE_DATABASE_README.md    # Esta documentación
```

## Solución de Problemas

### Errores Comunes:

1. **"No face embedding could be extracted"**
   - Solución: Verificar calidad de imagen y iluminación
   - Ajustar `enforce_detection=False` para imágenes difíciles

2. **"Database comparison too slow"**
   - Solución: Verificar tamaño de base de datos
   - Considerar implementar FAISS para >5000 rostros

3. **"Memory usage too high"**
   - Solución: Ajustar `CACHE_SIZE_LIMIT`
   - Implementar cache LRU más agresivo

### Contacto y Soporte:
Para problemas específicos o optimizaciones adicionales, revisar logs del sistema y métricas de rendimiento. 