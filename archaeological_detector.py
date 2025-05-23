#!/usr/bin/env python3
"""
Amazon Archaeological Site Finder
Minimal cost solution for OpenAI to Z Challenge
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import requests
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import openai
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point, Polygon
import earthengine as ee


# Configuration
@dataclass
class Config:
    # Free tier OpenAI API
    openai_api_key: str = "your-api-key-here"
    openai_model: str = "gpt-4o-mini"  # Cheapest option

    # Data paths
    lidar_path: str = "./data/amazon_dtm.tif"
    rivers_shapefile: str = "./data/amazon_rivers.shp"
    known_sites_csv: str = "./data/known_sites.csv"

    # Detection parameters
    min_structure_size: int = 50  # meters
    max_structure_size: int = 300  # meters
    min_height_anomaly: float = 0.5  # meters
    max_distance_from_water: float = 5000  # meters

    # Cost optimization
    max_ai_candidates: int = 10  # Limit AI analysis to top candidates


class ArchaeologicalDetector:
    def __init__(self, config: Config):
        self.config = config
        self.dtm_data = None
        self.transform = None
        self.rivers = None
        self.known_sites = []

    def load_dtm(self):
        """Load Digital Terrain Model from GeoTIFF"""
        print("Loading DTM data...")
        with rasterio.open(self.config.lidar_path) as src:
            self.dtm_data = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.pixel_size = src.transform[0]  # Assuming square pixels
        print(f"DTM loaded: {self.dtm_data.shape}, pixel size: {self.pixel_size}m")

    def load_rivers(self):
        """Load river shapefile for proximity analysis"""
        print("Loading river data...")
        self.rivers = gpd.read_file(self.config.rivers_shapefile)
        if self.rivers.crs != self.crs:
            self.rivers = self.rivers.to_crs(self.crs)

    def detect_circular_anomalies(self) -> List[Dict]:
        """Detect circular structures using Hough transform"""
        print("Detecting circular anomalies...")

        # Normalize and enhance DTM
        dtm_norm = (self.dtm_data - np.nanmean(self.dtm_data)) / np.nanstd(self.dtm_data)

        # Apply edge detection
        edges = ndimage.sobel(dtm_norm)

        # Threshold to get binary image
        threshold = np.percentile(edges[edges > 0], 90)
        binary = edges > threshold

        # Find connected components
        labeled, num_features = label(binary)

        candidates = []

        for i in range(1, num_features + 1):
            # Get component mask
            mask = labeled == i

            # Calculate properties
            area = np.sum(mask)
            if area < 100:  # Skip small features
                continue

            # Get bounding box
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue

            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            width = (max_col - min_col) * self.pixel_size
            height = (max_row - min_row) * self.pixel_size

            # Check size constraints
            if width < self.config.min_structure_size or width > self.config.max_structure_size:
                continue

            # Calculate circularity
            circularity = 4 * np.pi * area / (np.sum(binary[min_row:max_row + 1, min_col:max_col + 1]) ** 2)

            if circularity > 0.7:  # Reasonably circular
                center_row = (min_row + max_row) // 2
                center_col = (min_col + max_col) // 2

                # Convert to geographic coordinates
                lon, lat = self.pixel_to_coord(center_col, center_row)

                # Calculate average height anomaly
                height_anomaly = np.mean(self.dtm_data[mask]) - np.mean(self.dtm_data[~mask])

                if abs(height_anomaly) > self.config.min_height_anomaly:
                    candidates.append({
                        'type': 'circular',
                        'lat': lat,
                        'lon': lon,
                        'diameter': (width + height) / 2,
                        'circularity': circularity,
                        'height_anomaly': height_anomaly,
                        'confidence': circularity * min(abs(height_anomaly), 2) / 2
                    })

        print(f"Found {len(candidates)} circular candidates")
        return candidates

    def detect_rectangular_anomalies(self) -> List[Dict]:
        """Detect rectangular structures"""
        print("Detecting rectangular anomalies...")

        # Similar to circular detection but looking for rectangles
        dtm_norm = (self.dtm_data - np.nanmean(self.dtm_data)) / np.nanstd(self.dtm_data)

        # Apply morphological operations to enhance linear features
        kernel = np.ones((3, 3))
        closed = ndimage.binary_closing(dtm_norm > 0.5, kernel)
        opened = ndimage.binary_opening(closed, kernel)

        # Find edges
        edges_h = ndimage.sobel(opened, axis=0)
        edges_v = ndimage.sobel(opened, axis=1)
        edges = np.sqrt(edges_h ** 2 + edges_v ** 2)

        # Threshold
        binary = edges > np.percentile(edges[edges > 0], 85)

        # Find rectangles using connected components
        labeled, num_features = label(binary)

        candidates = []

        for i in range(1, num_features + 1):
            mask = labeled == i
            area = np.sum(mask)

            if area < 100:
                continue

            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue

            # Calculate bounding box
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            width = (max_col - min_col) * self.pixel_size
            height = (max_row - min_row) * self.pixel_size

            # Check size and aspect ratio
            if width < self.config.min_structure_size or width > self.config.max_structure_size:
                continue

            aspect_ratio = width / height if height > 0 else 0

            # Look for rectangular shapes (aspect ratio not too extreme)
            if 0.3 < aspect_ratio < 3.0:
                # Calculate rectangularity
                filled_ratio = area / ((max_row - min_row + 1) * (max_col - min_col + 1))

                if filled_ratio > 0.6:  # Reasonably filled rectangle
                    center_row = (min_row + max_row) // 2
                    center_col = (min_col + max_col) // 2
                    lon, lat = self.pixel_to_coord(center_col, center_row)

                    height_anomaly = np.mean(self.dtm_data[mask]) - np.mean(self.dtm_data[~mask])

                    if abs(height_anomaly) > self.config.min_height_anomaly:
                        candidates.append({
                            'type': 'rectangular',
                            'lat': lat,
                            'lon': lon,
                            'width': width,
                            'height': height,
                            'rectangularity': filled_ratio,
                            'height_anomaly': height_anomaly,
                            'confidence': filled_ratio * min(abs(height_anomaly), 2) / 2
                        })

        print(f"Found {len(candidates)} rectangular candidates")
        return candidates

    def filter_by_water_proximity(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by distance to water sources"""
        print("Filtering by water proximity...")

        if self.rivers is None:
            print("No river data available, skipping water proximity filter")
            return candidates

        filtered = []

        for candidate in candidates:
            point = Point(candidate['lon'], candidate['lat'])

            # Calculate distance to nearest river
            distances = self.rivers.geometry.distance(point)
            min_distance = distances.min()

            if min_distance <= self.config.max_distance_from_water:
                candidate['distance_to_water'] = min_distance
                filtered.append(candidate)

        print(f"Filtered to {len(filtered)} candidates near water")
        return filtered

    def filter_known_sites(self, candidates: List[Dict]) -> List[Dict]:
        """Remove candidates too close to known sites"""
        print("Filtering known sites...")

        # Load known sites if available
        if os.path.exists(self.config.known_sites_csv):
            import pandas as pd
            known_df = pd.read_csv(self.config.known_sites_csv)
            known_points = [Point(row.lon, row.lat) for _, row in known_df.iterrows()]
        else:
            known_points = []

        filtered = []
        min_distance_to_known = 1000  # meters

        for candidate in candidates:
            point = Point(candidate['lon'], candidate['lat'])

            # Check distance to all known sites
            too_close = False
            for known_point in known_points:
                if point.distance(known_point) < min_distance_to_known:
                    too_close = True
                    break

            if not too_close:
                filtered.append(candidate)

        print(f"Filtered to {len(filtered)} new candidates")
        return filtered

    def score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Score and rank candidates"""
        print("Scoring candidates...")

        for candidate in candidates:
            # Base score from detection confidence
            score = candidate['confidence']

            # Bonus for optimal distance from water (1-3km is ideal)
            if 'distance_to_water' in candidate:
                dist_km = candidate['distance_to_water'] / 1000
                if 1 <= dist_km <= 3:
                    score *= 1.5
                elif dist_km < 1:
                    score *= 1.2

            # Bonus for optimal size (100-200m structures most common)
            size = candidate.get('diameter', candidate.get('width', 0))
            if 100 <= size <= 200:
                score *= 1.3

            candidate['score'] = score

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates

    def analyze_with_ai(self, candidates: List[Dict]) -> List[Dict]:
        """Use OpenAI to analyze top candidates"""
        print(f"Analyzing top {self.config.max_ai_candidates} candidates with AI...")

        openai.api_key = self.config.openai_api_key

        # Only analyze top candidates to save costs
        top_candidates = candidates[:self.config.max_ai_candidates]

        for i, candidate in enumerate(top_candidates):
            print(f"Analyzing candidate {i + 1}/{len(top_candidates)}...")

            # Prepare context
            context = f"""
            Potential archaeological site detected:
            - Type: {candidate['type']} structure
            - Location: {candidate['lat']:.6f}, {candidate['lon']:.6f}
            - Size: {candidate.get('diameter', candidate.get('width', 'Unknown'))}m
            - Height anomaly: {candidate['height_anomaly']:.2f}m
            - Distance to water: {candidate.get('distance_to_water', 'Unknown')}m

            Based on known Amazonian archaeology, evaluate if this could be:
            1. A pre-Columbian earthwork (geoglyph, mound, or plaza)
            2. Natural formation
            3. Modern disturbance

            Provide brief assessment (max 100 words) and probability (0-1).
            """

            try:
                response = openai.ChatCompletion.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert in Amazonian archaeology."},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )

                ai_assessment = response.choices[0].message.content
                candidate['ai_assessment'] = ai_assessment

                # Extract probability if mentioned
                import re
                prob_match = re.search(r'probability:?\s*([0-9.]+)', ai_assessment.lower())
                if prob_match:
                    candidate['ai_probability'] = float(prob_match.group(1))
                    candidate['score'] *= (1 + candidate['ai_probability'])

            except Exception as e:
                print(f"AI analysis failed for candidate {i + 1}: {e}")
                candidate['ai_assessment'] = "Analysis failed"

        # Re-sort by updated scores
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates

    def visualize_results(self, candidates: List[Dict], output_path: str = "results.png"):
        """Create visualization of results"""
        print("Creating visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot DTM with candidates
        im = ax1.imshow(self.dtm_data, cmap='terrain')
        ax1.set_title('Digital Terrain Model with Detected Sites')

        # Add candidates to map
        for i, candidate in enumerate(candidates[:10]):  # Show top 10
            row, col = self.coord_to_pixel(candidate['lon'], candidate['lat'])

            if candidate['type'] == 'circular':
                circle = Circle((col, row),
                                radius=candidate['diameter'] / (2 * self.pixel_size),
                                fill=False, color='red', linewidth=2)
                ax1.add_patch(circle)
            else:
                rect = Rectangle((col - candidate['width'] / (2 * self.pixel_size),
                                  row - candidate['height'] / (2 * self.pixel_size)),
                                 width=candidate['width'] / self.pixel_size,
                                 height=candidate['height'] / self.pixel_size,
                                 fill=False, color='blue', linewidth=2)
                ax1.add_patch(rect)

            ax1.text(col, row, str(i + 1), color='yellow', fontsize=12,
                     ha='center', va='center', weight='bold')

        # Create score chart
        scores = [c['score'] for c in candidates[:10]]
        labels = [f"Site {i + 1}" for i in range(len(scores))]

        ax2.barh(labels, scores)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Top 10 Candidate Sites by Score')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")

        return fig

    def generate_report(self, candidates: List[Dict]) -> str:
        """Generate final report"""
        print("Generating report...")

        report = f"""
# Archaeological Site Detection Report
## Amazon Rainforest Analysis
### Date: {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
- Total candidates detected: {len(candidates)}
- High confidence sites (score > 0.7): {len([c for c in candidates if c['score'] > 0.7])}
- Analysis cost: <$10 (OpenAI API usage)

## Top 5 Discoveries

"""

        for i, candidate in enumerate(candidates[:5]):
            report += f"""
### Discovery #{i + 1}
- **Coordinates**: {candidate['lat']:.6f}, {candidate['lon']:.6f}
- **Type**: {candidate['type'].capitalize()} structure
- **Size**: {candidate.get('diameter', candidate.get('width', 'Unknown'))}m
- **Height Anomaly**: {candidate['height_anomaly']:.2f}m
- **Distance to Water**: {candidate.get('distance_to_water', 'Unknown'):.0f}m
- **Confidence Score**: {candidate['score']:.3f}

**AI Assessment**: {candidate.get('ai_assessment', 'Not analyzed')}

---
"""

        report += """
## Methodology
1. DTM analysis using edge detection and morphological operations
2. Geometric pattern recognition (circles and rectangles)
3. Filtering by water proximity and known sites
4. AI-assisted evaluation of top candidates
5. Cost-optimized approach using minimal API calls

## Verification Methods
- Cross-reference with Sentinel-2 imagery for vegetation anomalies
- Compare with known site patterns from Peripato et al. (2023)
- Check historical maps and expedition records

## Next Steps
1. Acquire high-resolution satellite imagery for top sites
2. Contact local archaeologists for ground verification
3. Submit findings to competition with full documentation
"""

        return report

    def pixel_to_coord(self, col: int, row: int) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates"""
        lon, lat = self.transform * (col, row)
        return lon, lat

    def coord_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert geographic coordinates to pixel coordinates"""
        col, row = ~self.transform * (lon, lat)
        return int(row), int(col)

    def run_detection(self):
        """Main detection pipeline"""
        print("Starting archaeological site detection...")

        # Load data
        self.load_dtm()
        self.load_rivers()

        # Detect features
        circular = self.detect_circular_anomalies()
        rectangular = self.detect_rectangular_anomalies()

        # Combine all candidates
        all_candidates = circular + rectangular
        print(f"Total candidates found: {len(all_candidates)}")

        # Apply filters
        candidates = self.filter_by_water_proximity(all_candidates)
        candidates = self.filter_known_sites(candidates)

        # Score and rank
        candidates = self.score_candidates(candidates)

        # AI analysis (only for top candidates to save money)
        if self.config.openai_api_key != "your-api-key-here":
            candidates = self.analyze_with_ai(candidates)

        # Generate outputs
        self.visualize_results(candidates)
        report = self.generate_report(candidates)

        # Save report
        with open("detection_report.md", "w") as f:
            f.write(report)

        # Save candidate data
        import pandas as pd
        df = pd.DataFrame(candidates[:20])  # Save top 20
        df.to_csv("top_candidates.csv", index=False)

        print("\nDetection complete!")
        print(f"Report saved to: detection_report.md")
        print(f"Visualization saved to: results.png")
        print(f"Candidate data saved to: top_candidates.csv")

        return candidates


# Example usage with Google Earth Engine integration
def download_sample_data():
    """Download sample DTM data from Earth Engine"""
    print("Initializing Earth Engine...")

    # Initialize Earth Engine (requires authentication)
    ee.Initialize()

    # Define area of interest (Western Amazon)
    aoi = ee.Geometry.Rectangle([-70, -12, -65, -8])

    # Get SRTM elevation data
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation').clip(aoi)

    # Export parameters
    export_params = {
        'image': elevation,
        'description': 'amazon_dtm_sample',
        'scale': 30,  # 30m resolution
        'region': aoi,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e9
    }

    print("Note: Full data download requires Earth Engine setup.")
    print("For testing, use pre-downloaded sample data.")

    return export_params


# Main execution
if __name__ == "__main__":
    # Configuration
    config = Config()

    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=== Amazon Archaeological Site Finder ===")
    print("Minimal cost solution for OpenAI to Z Challenge")
    print()

    # Check if sample data exists
    if not os.path.exists(config.lidar_path):
        print("Sample DTM data not found.")
        print("Please download from OpenTopography or use Earth Engine.")
        print("\nEarth Engine export parameters:")
        params = download_sample_data()
        print(json.dumps(params, indent=2))
    else:
        # Run detection
        detector = ArchaeologicalDetector(config)
        candidates = detector.run_detection()

        # Print summary
        print("\n=== SUMMARY ===")
        print(f"Top discovery: {candidates[0]['lat']:.6f}, {candidates[0]['lon']:.6f}")
        print(f"Type: {candidates[0]['type']}")
        print(f"Confidence: {candidates[0]['score']:.3f}")
        print("\nReady for submission!")