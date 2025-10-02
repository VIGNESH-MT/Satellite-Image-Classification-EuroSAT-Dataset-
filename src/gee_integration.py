"""
Google Earth Engine integration for land usage visualization.
Fetches satellite imagery and provides land usage analysis.
"""

import ee
import folium
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image

from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class GEELandUsageAnalyzer:
    """
    Google Earth Engine integration for land usage analysis and visualization.
    """
    
    def __init__(self, service_account_key: str = None):
        """
        Initialize the GEE analyzer.
        
        Args:
            service_account_key: Path to GEE service account key file
        """
        self.service_account_key = service_account_key or GEE_SERVICE_ACCOUNT_KEY
        self.is_authenticated = False
        self.land_cover_palette = [
            '#419bdf', '#397d49', '#88b053', '#7a87c6',
            '#e49635', '#dfc35a', '#c4281b', '#a59b8f',
            '#88b053', '#23cce2'
        ]
        
    def authenticate(self):
        """
        Authenticate with Google Earth Engine.
        """
        try:
            if self.service_account_key and Path(self.service_account_key).exists():
                # Service account authentication
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Will be read from key file
                    key_file=self.service_account_key
                )
                ee.Initialize(credentials)
                logger.info("Authenticated with GEE using service account")
            else:
                # Try to use existing authentication
                ee.Initialize()
                logger.info("Authenticated with GEE using existing credentials")
                
            self.is_authenticated = True
            
        except Exception as e:
            logger.warning(f"GEE authentication failed: {str(e)}")
            logger.info("Please authenticate with: earthengine authenticate")
            self.is_authenticated = False
            
    def get_satellite_image(self, coordinates: Tuple[float, float], 
                          zoom_level: int = 13, 
                          start_date: str = None,
                          end_date: str = None) -> Optional[ee.Image]:
        """
        Get satellite image for given coordinates.
        
        Args:
            coordinates: (latitude, longitude) tuple
            zoom_level: Zoom level for the image
            start_date: Start date for image collection (YYYY-MM-DD)
            end_date: End date for image collection (YYYY-MM-DD)
            
        Returns:
            Earth Engine Image object
        """
        if not self.is_authenticated:
            logger.error("Not authenticated with GEE")
            return None
            
        try:
            lat, lon = coordinates
            
            # Define area of interest
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(1000)  # 1km buffer
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Get Sentinel-2 imagery
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            # Get median composite
            image = collection.median().clip(region)
            
            logger.info(f"Retrieved satellite image for coordinates {coordinates}")
            return image
            
        except Exception as e:
            logger.error(f"Error retrieving satellite image: {str(e)}")
            return None
            
    def get_land_cover_data(self, coordinates: Tuple[float, float], 
                           buffer_size: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Get land cover data for given coordinates.
        
        Args:
            coordinates: (latitude, longitude) tuple
            buffer_size: Buffer size in meters
            
        Returns:
            Dictionary with land cover information
        """
        if not self.is_authenticated:
            logger.error("Not authenticated with GEE")
            return None
            
        try:
            lat, lon = coordinates
            
            # Define area of interest
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(buffer_size)
            
            # Get ESA WorldCover land cover data
            land_cover = ee.ImageCollection('ESA/WorldCover/v100').first()
            
            # Clip to region
            land_cover_clipped = land_cover.clip(region)
            
            # Calculate land cover statistics
            stats = land_cover_clipped.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=region,
                scale=10,
                maxPixels=1e9
            )
            
            # Get the histogram
            histogram = stats.getInfo()['Map']
            
            # Map land cover codes to names
            land_cover_map = {
                '10': 'Tree cover',
                '20': 'Shrubland',
                '30': 'Grassland',
                '40': 'Cropland',
                '50': 'Built-up',
                '60': 'Bare/sparse vegetation',
                '70': 'Snow and ice',
                '80': 'Permanent water bodies',
                '90': 'Herbaceous wetland',
                '95': 'Mangroves',
                '100': 'Moss and lichen'
            }
            
            # Convert to readable format
            land_cover_stats = {}
            total_pixels = sum(histogram.values())
            
            for code, count in histogram.items():
                land_type = land_cover_map.get(code, f'Unknown ({code})')
                percentage = (count / total_pixels) * 100
                land_cover_stats[land_type] = {
                    'pixels': count,
                    'percentage': round(percentage, 2)
                }
            
            result = {
                'coordinates': coordinates,
                'buffer_size_m': buffer_size,
                'total_pixels': total_pixels,
                'land_cover_distribution': land_cover_stats,
                'dominant_land_cover': max(land_cover_stats.items(), 
                                         key=lambda x: x[1]['percentage'])[0]
            }
            
            logger.info(f"Retrieved land cover data for coordinates {coordinates}")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving land cover data: {str(e)}")
            return None
            
    def create_land_usage_map(self, coordinates: Tuple[float, float], 
                             zoom_level: int = 13) -> Optional[folium.Map]:
        """
        Create an interactive map showing land usage.
        
        Args:
            coordinates: (latitude, longitude) tuple
            zoom_level: Zoom level for the map
            
        Returns:
            Folium map object
        """
        try:
            lat, lon = coordinates
            
            # Create base map
            m = folium.Map(location=[lat, lon], zoom_start=zoom_level)
            
            # Add marker for the point of interest
            folium.Marker(
                [lat, lon],
                popup=f"Location: {lat:.4f}, {lon:.4f}",
                tooltip="Click for coordinates",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            if self.is_authenticated:
                try:
                    # Get land cover data
                    land_cover_data = self.get_land_cover_data(coordinates)
                    
                    if land_cover_data:
                        # Add land cover information to popup
                        popup_html = f"""
                        <div style="width: 300px;">
                            <h4>Land Cover Analysis</h4>
                            <p><b>Coordinates:</b> {lat:.4f}, {lon:.4f}</p>
                            <p><b>Dominant Land Cover:</b> {land_cover_data['dominant_land_cover']}</p>
                            <h5>Distribution:</h5>
                            <ul>
                        """
                        
                        for land_type, stats in land_cover_data['land_cover_distribution'].items():
                            if stats['percentage'] > 1:  # Only show types with >1%
                                popup_html += f"<li>{land_type}: {stats['percentage']:.1f}%</li>"
                        
                        popup_html += "</ul></div>"
                        
                        # Add circle showing analysis area
                        folium.Circle(
                            location=[lat, lon],
                            radius=1000,  # 1km radius
                            popup=folium.Popup(popup_html, max_width=350),
                            color='blue',
                            fill=True,
                            fillOpacity=0.2
                        ).add_to(m)
                        
                except Exception as e:
                    logger.warning(f"Could not add GEE data to map: {str(e)}")
            
            # Add different tile layers
            folium.TileLayer('OpenStreetMap').add_to(m)
            folium.TileLayer('Stamen Terrain').add_to(m)
            folium.TileLayer('Stamen Toner').add_to(m)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            logger.info(f"Created land usage map for coordinates {coordinates}")
            return m
            
        except Exception as e:
            logger.error(f"Error creating land usage map: {str(e)}")
            return None
            
    def analyze_region_for_classification(self, coordinates: Tuple[float, float], 
                                        predicted_class: str) -> Dict[str, Any]:
        """
        Analyze a region and compare with predicted classification.
        
        Args:
            coordinates: (latitude, longitude) tuple
            predicted_class: Predicted land use class
            
        Returns:
            Analysis results
        """
        analysis = {
            'coordinates': coordinates,
            'predicted_class': predicted_class,
            'gee_analysis': None,
            'confidence_score': 0.0,
            'recommendations': []
        }
        
        if not self.is_authenticated:
            analysis['recommendations'].append(
                "Google Earth Engine not authenticated. Limited analysis available."
            )
            return analysis
            
        try:
            # Get land cover data
            land_cover_data = self.get_land_cover_data(coordinates)
            
            if land_cover_data:
                analysis['gee_analysis'] = land_cover_data
                
                # Map EuroSAT classes to GEE land cover types
                class_mapping = {
                    'AnnualCrop': ['Cropland'],
                    'Forest': ['Tree cover'],
                    'HerbaceousVegetation': ['Grassland', 'Herbaceous wetland'],
                    'Highway': ['Built-up'],
                    'Industrial': ['Built-up'],
                    'Pasture': ['Grassland'],
                    'PermanentCrop': ['Cropland', 'Tree cover'],
                    'Residential': ['Built-up'],
                    'River': ['Permanent water bodies'],
                    'SeaLake': ['Permanent water bodies']
                }
                
                # Calculate confidence based on land cover match
                expected_types = class_mapping.get(predicted_class, [])
                total_expected_percentage = 0
                
                for land_type, stats in land_cover_data['land_cover_distribution'].items():
                    if any(expected in land_type for expected in expected_types):
                        total_expected_percentage += stats['percentage']
                
                analysis['confidence_score'] = min(total_expected_percentage / 100.0, 1.0)
                
                # Generate recommendations
                if analysis['confidence_score'] > 0.7:
                    analysis['recommendations'].append(
                        f"High confidence: GEE data strongly supports {predicted_class} classification"
                    )
                elif analysis['confidence_score'] > 0.4:
                    analysis['recommendations'].append(
                        f"Medium confidence: GEE data partially supports {predicted_class} classification"
                    )
                else:
                    analysis['recommendations'].append(
                        f"Low confidence: GEE data suggests different land use than {predicted_class}"
                    )
                    dominant = land_cover_data['dominant_land_cover']
                    analysis['recommendations'].append(
                        f"Dominant land cover according to GEE: {dominant}"
                    )
                    
        except Exception as e:
            logger.error(f"Error in region analysis: {str(e)}")
            analysis['recommendations'].append(f"Analysis error: {str(e)}")
            
        return analysis


def create_demo_map(coordinates: Tuple[float, float] = (52.5200, 13.4050)) -> str:
    """
    Create a demo map for testing (Berlin coordinates by default).
    
    Args:
        coordinates: (latitude, longitude) tuple
        
    Returns:
        Path to saved HTML map
    """
    analyzer = GEELandUsageAnalyzer()
    analyzer.authenticate()
    
    # Create map
    map_obj = analyzer.create_land_usage_map(coordinates)
    
    if map_obj:
        # Save map
        map_path = STATIC_DIR / "demo_map.html"
        map_obj.save(str(map_path))
        logger.info(f"Demo map saved to {map_path}")
        return str(map_path)
    else:
        logger.error("Failed to create demo map")
        return None


def setup_gee_authentication():
    """
    Setup instructions for GEE authentication.
    """
    print("Google Earth Engine Setup Instructions:")
    print("=" * 50)
    print("1. Install the Earth Engine Python API:")
    print("   pip install earthengine-api")
    print()
    print("2. Authenticate with Earth Engine:")
    print("   earthengine authenticate")
    print()
    print("3. For service account authentication:")
    print("   - Create a service account in Google Cloud Console")
    print("   - Download the JSON key file")
    print("   - Set the GEE_SERVICE_ACCOUNT_KEY environment variable")
    print()
    print("4. Test authentication:")
    print("   python -c \"import ee; ee.Initialize(); print('Success!')\"")
    print()


if __name__ == "__main__":
    # Demo usage
    setup_gee_authentication()
    
    # Test coordinates (Berlin, Germany)
    test_coords = (52.5200, 13.4050)
    
    analyzer = GEELandUsageAnalyzer()
    analyzer.authenticate()
    
    if analyzer.is_authenticated:
        # Test land cover analysis
        land_cover = analyzer.get_land_cover_data(test_coords)
        if land_cover:
            print(f"Land cover analysis for {test_coords}:")
            print(f"Dominant land cover: {land_cover['dominant_land_cover']}")
            
        # Create demo map
        demo_map_path = create_demo_map(test_coords)
        if demo_map_path:
            print(f"Demo map created: {demo_map_path}")
    else:
        print("GEE authentication required for full functionality")
