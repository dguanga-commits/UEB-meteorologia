#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para descargar y filtrar los límites cantonales de la provincia de Bolívar, Ecuador.
Genera el archivo local 'bolivar_cantones.geojson' para uso en el dashboard.
"""

import urllib.request
import json
import os

def download_and_filter_geojson():
    # URL de un Gist con cantones de Ecuador (formato estandarizado con DPA_PROVIN y DPA_DESCAN)
    url_geojson = "https://gist.githubusercontent.com/emamut/25912e117ab46fa00a63c6e890575201/raw/cantones-ecuador.geojson"
    output_filename = "bolivar_cantones.geojson"
    
    print("=== DESCARGANDO Y PROCESANDO GEOJSON DE BOLÍVAR ===")
    
    data = None
    try:
        print(f"Descargando GeoJSON desde la fuente: {url_geojson} ...")
        # Configurar un User-Agent para evitar posibles bloqueos
        req = urllib.request.Request(
            url_geojson, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            print("✓ Descarga completada con éxito.")
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return False

    if not data:
        print("❌ El archivo descargado está vacío.")
        return False
        
    print(f"Total de entidades (features) en el GeoJSON: {len(data.get('features', []))}")
    
    # Filtrar solo cantones de la provincia de Bolívar (código de provincia "02")
    bolivar_features = []
    
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        prov_code = props.get('DPA_PROVIN', '')
        prov_name = props.get('DPA_DESPRO', '').strip().upper()
        canton_name = props.get('DPA_DESCAN', '').strip().upper()
        canton_code = props.get('DPA_CANTON', '')
        
        # Filtros de coincidencia para Bolívar (código "02")
        if str(prov_code) == "02" or "BOLIVAR" in prov_name or "BOLÍVAR" in prov_name:
            # Normalizar nombres de cantones
            if "GUARANDA" in canton_name:
                props['name'] = "Guaranda"
            elif "SAN MIGUEL" in canton_name:
                props['name'] = "San Miguel"
            elif "CHIMBO" in canton_name:
                props['name'] = "Chimbo"
            elif "CHILLANES" in canton_name:
                props['name'] = "Chillanes"
            elif "CALUMA" in canton_name:
                props['name'] = "Caluma"
            elif "ECHEANDIA" in canton_name or "ECHEANDÍA" in canton_name:
                props['name'] = "Echeandía"
            elif "LAS NAVES" in canton_name:
                props['name'] = "Las Naves"
            else:
                props['name'] = canton_name.title()
                
            # Limpiar el feature guardando solo lo necesario para el mapa
            feature['properties'] = {
                'name': props['name'],
                'code': canton_code,
                'province': 'Bolívar'
            }
            bolivar_features.append(feature)
            print(f"  - Cantón detectado: {props['name']} (Código: {canton_code})")
            
    if not bolivar_features:
        print("❌ Error: No se encontraron cantones para la provincia de Bolívar.")
        return False
        
    # Crear un nuevo objeto GeoJSON con los datos filtrados
    bolivar_geojson = {
        "type": "FeatureCollection",
        "features": bolivar_features
    }
    
    # Guardar en archivo local
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(bolivar_geojson, f, ensure_ascii=False, indent=2)
        
    print(f"✓ Archivo GeoJSON guardado con éxito: '{output_filename}' ({len(bolivar_features)} cantones)")
    return True

if __name__ == "__main__":
    download_and_filter_geojson()
