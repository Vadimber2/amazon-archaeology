#!/usr/bin/env python3
"""
Data Downloader for Amazon Archaeology Project
Downloads free LiDAR and satellite data
"""

import os
import requests
import zipfile
import geopandas as gpd
from pyproj import Transformer
import earthengine as ee
import geemap


class DataDownloader:
    def __init__(self, output_dir="./data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_opentopography_lidar(self, bbox=None):
        """
        Download LiDAR data from OpenTopography
        bbox: [west, south, east, north] in decimal degrees
        """
        if bbox is None:
            # Default area: Western Amazon
            bbox = [-70, -12, -65, -8]

        print("Downloading LiDAR from OpenTopography...")

        # OpenTopography API endpoint
        base_url = "https://portal.opentopography.org/API/otCatalog"

        params = {
            'west': bbox[0],
            'south': bbox[1],
            'east': bbox[2],
            'north': bbox[3],
            'outputFormat': 'json'
        }

        # Get available datasets
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            datasets = response.json()
            print(f"Found {len(datasets)} LiDAR datasets")

            # Download first available dataset
            for dataset in datasets:
                if 'Brazil' in dataset.get('title', '') or 'Amazon' in dataset.get('title', ''):
                    print(f"Downloading: {dataset['title']}")
                    # Note: Actual download requires API key
                    print("Note: OpenTopography requires registration for downloads")
                    print(f"Dataset ID: {dataset['id']}")
                    break
        else:
            print("Failed to query OpenTopography")

    def download_sentinel2_composite(self, bbox=None, start_date='2024-01-01', end_date='2024-12-31'):
        """
        Create and download Sentinel-2 composite using Earth Engine
        """
        if bbox is None:
            bbox = [-70, -12, -65, -8]

        print("Creating Sentinel-2 composite...")

        # Initialize Earth Engine
        try:
            ee.Initialize()
        except:
            print("Earth Engine not authenticated. Run: earthengine authenticate")
            return

        # Define area of interest
        aoi = ee.Geometry.Rectangle(bbox)

        # Load Sentinel-2 collection
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        # Create median composite
        composite = s2.median().clip(aoi)

        # Select bands for visualization
        rgb = composite.select(['B4', 'B3', 'B2'])

        # Export parameters
        export_params = {
            'image': rgb,
            'description': 'amazon_sentinel2_composite',
            'scale': 10,
            'region': aoi,
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e9
        }

        print("Sentinel-2 composite ready for export")
        return export_params

    def download_srtm_elevation(self, bbox=None):
        """
        Download SRTM elevation data
        """
        if bbox is None:
            bbox = [-70, -12, -65, -8]

        print("Downloading SRTM elevation data...")

        try:
            ee.Initialize()
        except:
            print("Earth Engine not authenticated")
            return

        # Define AOI
        aoi = ee.Geometry.Rectangle(bbox)

        # Load SRTM
        srtm = ee.Image('USGS/SRTMGL1_003')
        elevation = srtm.select('elevation').clip(aoi)

        # Calculate slope for better feature detection
        slope = ee.Terrain.slope(elevation)

        # Stack elevation and slope
        terrain = elevation.addBands(slope)

        # Use geemap for easy download
        import geemap

        # Create map
        Map = geemap.Map()
        Map.centerObject(aoi, 8)
        Map.addLayer(elevation, {'min': 0, 'max': 500}, 'Elevation')
        Map.addLayer(slope, {'min': 0, 'max': 30}, 'Slope')

        # Download
        filename = os.path.join(self.output_dir, 'srtm_elevation.tif')
        geemap.ee_export_image(elevation, filename=filename, scale=30, region=aoi)

        print(f"SRTM saved to: {filename}")

    def download_rivers(self, bbox=None):
        """
        Download river network from OpenStreetMap
        """
        if bbox is None:
            bbox = [-70, -12, -65, -8]

        print("Downloading river data from OSM...")

        # Overpass API query
        overpass_url = "http://overpass-api.de/api/interpreter"

        # Query for rivers and streams
        query = f"""
        [out:json];
        (
          way["waterway"="river"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way["waterway"="stream"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out geom;
        """

        response = requests.get(overpass_url, params={'data': query})

        if response.status_code == 200:
            data = response.json()

            # Convert to GeoDataFrame
            features = []
            for element in data['elements']:
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    features.append({
                        'geometry': gpd.points_from_xy(*zip(*coords)),
                        'name': element.get('tags', {}).get('name', 'Unknown'),
                        'type': element.get('tags', {}).get('waterway', 'river')
                    })

            if features:
                gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
                filename = os.path.join(self.output_dir, 'amazon_rivers.shp')
                gdf.to_file(filename)
                print(f"Rivers saved to: {filename}")
            else:
                print("No river data found")
        else:
            print("Failed to download river data")

    def download_known_sites(self):
        """
        Download known archaeological sites from various sources
        """
        print("Compiling known archaeological sites...")

        # Sample known sites (from literature)
        known_sites = [
            # Geoglyphs from Acre state
            {'name': 'Fazenda Colorada', 'lat': -9.8275, 'lon': -67.5369, 'type': 'geoglyph'},
            {'name': 'Fazenda Atlantica', 'lat': -10.1833, 'lon': -67.5333, 'type': 'geoglyph'},
            {'name': 'Jacó Sá', 'lat': -10.0667, 'lon': -67.5167, 'type': 'geoglyph'},

            # Mound sites from Llanos de Mojos
            {'name': 'Bella Vista', 'lat': -14.8333, 'lon': -64.9167, 'type': 'mound'},
            {'name': 'El Cerro', 'lat': -14.7500, 'lon': -65.0000, 'type': 'mound'},

            # Terra preta sites
            {'name': 'Hatahara', 'lat': -3.2833, 'lon': -60.2000, 'type': 'terra_preta'},
            {'name': 'Açutuba', 'lat': -3.1000, 'lon': -60.3500, 'type': 'terra_preta'},
        ]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            known_sites,
            geometry=gpd.points_from_xy(
                [s['lon'] for s in known_sites],
                [s['lat'] for s in known_sites]
            ),
            crs='EPSG:4326'
        )

        # Save as CSV and shapefile
        csv_file = os.path.join(self.output_dir, 'known_sites.csv')
        shp_file = os.path.join(self.output_dir, 'known_sites.shp')

        gdf.to_csv(csv_file, index=False)
        gdf.to_file(shp_file)

        print(f"Known sites saved to: {csv_file}")

    def create_sample_dtm(self):
        """
        Create a sample DTM with synthetic archaeological features for testing
        """
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        print("Creating sample DTM for testing...")

        # Create synthetic terrain
        size = 1000
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)

        # Base terrain (gentle hills)
        terrain = 100 + 5 * np.sin(X / 2) * np.cos(Y / 2) + np.random.normal(0, 0.5, (size, size))

        # Add circular mound
        center1 = (300, 400)
        radius1 = 50
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center1[0]) ** 2 + (j - center1[1]) ** 2)
                if dist < radius1:
                    terrain[i, j] += 2 * (1 - dist / radius1)

        # Add rectangular platform
        rect_start = (600, 600)
        rect_size = (100, 150)
        terrain[rect_start[0]:rect_start[0] + rect_size[0],
        rect_start[1]:rect_start[1] + rect_size[1]] += 1.5

        # Add ring ditch (geoglyph)
        center2 = (700, 200)
        radius2_outer = 80
        radius2_inner = 60
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center2[0]) ** 2 + (j - center2[1]) ** 2)
                if radius2_inner < dist < radius2_outer:
                    terrain[i, j] -= 1.0

        # Define bounds (Western Amazon)
        bounds = (-69, -11, -68.9, -10.9)
        transform = from_bounds(*bounds, size, size)

        # Save as GeoTIFF
        filename = os.path.join(self.output_dir, 'amazon_dtm.tif')

        with rasterio.open(
                filename,
                'w',
                driver='GTiff',
                height=size,
                width=size,
                count=1,
                dtype=terrain.dtype,
                crs='EPSG:4326',
                transform=transform,
        ) as dst:
            dst.write(terrain, 1)

        print(f"Sample DTM saved to: {filename}")
        print("Contains 3 synthetic features: circular mound, rectangular platform, ring ditch")


# Quick download script
def quick_setup():
    """Quick setup with sample data for testing"""
    downloader = DataDownloader()

    print("=== Quick Setup for Amazon Archaeology ===")
    print()

    # Create sample data
    downloader.create_sample_dtm()
    downloader.download_known_sites()

    print("\nSample data created!")
    print("To download real data:")
    print("1. Set up Earth Engine: earthengine authenticate")
    print("2. Register at OpenTopography.org for LiDAR access")
    print("3. Run full download functions")


if __name__ == "__main__":
    quick_setup()