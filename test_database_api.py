#!/usr/bin/env python3
"""
Script de prueba para los nuevos endpoints de base de datos facial
"""

import requests
import base64
import json
import time
from pathlib import Path

# Configuración
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # Cambiar por una imagen real

def encode_image_to_base64(image_path: str) -> str:
    """Convierte una imagen a base64"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"Error leyendo imagen {image_path}: {e}")
        return None

def test_database_stats():
    """Prueba el endpoint de estadísticas"""
    print("\n=== PROBANDO ESTADÍSTICAS DE BASE DE DATOS ===")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/photo/database-stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Estadísticas obtenidas:")
            print(f"   - Total embeddings: {stats['stats']['total_embeddings']}")
            print(f"   - Archivo DB: {stats['stats']['storage_path']}")
            print(f"   - Usuarios: {stats['stats']['users'][:5]}...")  # Primeros 5
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error en petición: {e}")

def test_add_face(user_id: str, image_path: str):
    """Prueba agregar un rostro a la base de datos"""
    print(f"\n=== PROBANDO AGREGAR ROSTRO: {user_id} ===")
    
    image_b64 = encode_image_to_base64(image_path)
    if not image_b64:
        print("❌ No se pudo leer la imagen")
        return False
    
    try:
        payload = {
            "user_id": user_id,
            "image": image_b64,
            "metadata": {
                "test_user": True,
                "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "model_name": "Facenet",
            "enforce_detection": True,
            "detector_backend": "ssd"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/photo/add-face-to-database", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Rostro agregado exitosamente:")
            print(f"   - Usuario: {result['user_id']}")
            print(f"   - Tiempo procesamiento: {result['processing_time']:.2f}s")
            print(f"   - Total en DB: {result['total_faces_in_database']}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error en petición: {e}")
        return False

def test_compare_with_database(image_path: str, expected_user: str = None):
    """Prueba la comparación con la base de datos"""
    print(f"\n=== PROBANDO COMPARACIÓN CON BASE DE DATOS ===")
    
    image_b64 = encode_image_to_base64(image_path)
    if not image_b64:
        print("❌ No se pudo leer la imagen")
        return
    
    try:
        payload = {
            "image": image_b64,
            "user_id": "test_comparison",
            "threshold": 0.6,
            "model_name": "Facenet",
            "enforce_detection": False,
            "detector_backend": "ssd"
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/api/photo/compare-with-database", json=payload)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Comparación completada:")
            print(f"   - Tiempo total petición: {request_time:.2f}s")
            print(f"   - Tiempo procesamiento: {result['processing_time']:.2f}s")
            print(f"   - Rostros comparados: {result['total_faces_compared']}")
            print(f"   - Acceso permitido: {'SÍ' if result['access_granted'] else 'NO'}")
            
            if result['match_found']:
                best_match = result['best_match']
                print(f"   - Mejor coincidencia:")
                print(f"     * Usuario: {best_match['user_id']}")
                print(f"     * Similitud: {best_match['similarity_score']:.3f}")
                print(f"     * Distancia: {best_match['distance']:.3f}")
                
                if expected_user and best_match['user_id'] == expected_user:
                    print(f"   ✅ Coincidencia correcta con usuario esperado")
                elif expected_user:
                    print(f"   ⚠️  Usuario diferente al esperado ({expected_user})")
                
                if len(result['all_matches']) > 1:
                    print(f"   - Otras coincidencias: {len(result['all_matches'])-1}")
            else:
                print(f"   - No se encontraron coincidencias")
                
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error en petición: {e}")

def test_remove_face(user_id: str):
    """Prueba eliminar un rostro de la base de datos"""
    print(f"\n=== PROBANDO ELIMINAR ROSTRO: {user_id} ===")
    
    try:
        response = requests.delete(f"{API_BASE_URL}/api/photo/remove-face/{user_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Rostro eliminado exitosamente:")
            print(f"   - Usuario: {user_id}")
            print(f"   - Total en DB: {result['total_faces_in_database']}")
        elif response.status_code == 404:
            print(f"⚠️  Usuario {user_id} no encontrado en la base de datos")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Error en petición: {e}")

def main():
    """Ejecuta todas las pruebas"""
    print("🧪 INICIANDO PRUEBAS DE API DE BASE DE DATOS FACIAL")
    print(f"🌐 URL Base: {API_BASE_URL}")
    
    # Verificar si el servidor está corriendo
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code != 200:
            print("❌ El servidor no parece estar corriendo. Ejecuta: uvicorn app:app --reload")
            return
    except:
        print("❌ No se puede conectar al servidor. Verificar que esté corriendo en el puerto 8000")
        return
    
    print("✅ Servidor detectado")
    
    # Prueba 1: Estadísticas iniciales
    test_database_stats()
    
    # Si no tienes una imagen de prueba, usar una imagen sintética
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"\n⚠️  Imagen de prueba {TEST_IMAGE_PATH} no encontrada")
        print("   Para pruebas completas, proporciona una imagen de rostro válida")
        print("   Por ahora solo se probarán endpoints que no requieren imagen")
        return
    
    # Prueba 2: Agregar rostro de prueba
    test_user_id = "test_user_001"
    if test_add_face(test_user_id, TEST_IMAGE_PATH):
        
        # Prueba 3: Estadísticas después de agregar
        test_database_stats()
        
        # Prueba 4: Comparación con la misma imagen
        test_compare_with_database(TEST_IMAGE_PATH, test_user_id)
        
        # Prueba 5: Eliminar rostro de prueba
        test_remove_face(test_user_id)
        
        # Prueba 6: Estadísticas finales
        test_database_stats()
    
    print("\n🎉 PRUEBAS COMPLETADAS")

if __name__ == "__main__":
    main() 