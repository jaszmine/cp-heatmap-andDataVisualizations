import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import time
import requests

# PSEUDOCODE STEPS TO FOLLOW
#   1. SETUP & DEPENDENCY CHECK
#   2. PRINT DATA SUMMARY
#   3. DATA ENHANCEMENT PIPELINE
#   4. CREATE MAIN HEATMAP (disaster_heatmap.html)
#   5. CREATE SIMPLE HEATMAP (disaster_heatmap_simple.html)
#   6. OUTPUT RESULTS

# ===================================
CSV_FILE_PATH = "114records-be_extracted_info_output_rows.csv"
# ===================================

# Initialize geocoder
geolocator = Nominatim(user_agent="disaster_heatmap_app")

def geocode_location(location_name):
    """
    Convert location name to coordinates using geopy
    """
    if pd.isna(location_name) or location_name == '' or location_name == 'null':
        return None, None
    
    try:
        # Add delay to respect rate limits
        time.sleep(0.1)
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Geocoding error for '{location_name}': {e}")
    
    return None, None

def get_coordinates_from_nominatim(location_name):
    """
    Alternative method using direct Nominatim API
    """
    if pd.isna(location_name) or location_name == '' or location_name == 'null':
        return None, None
    
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'DisasterHeatmapApp/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except Exception as e:
        print(f"API geocoding error for '{location_name}': {e}")
    
    return None, None

def enhance_data_with_coordinates(df):
    """
    Enhance the dataframe by generating missing coordinates from location_mentioned
    """
    enhanced_df = df.copy()
    enhanced_count = 0
    
    print("Enhancing data with missing coordinates...")
    
    for idx, row in enhanced_df.iterrows():
        # Check if coordinates are missing or invalid
        lat_missing = pd.isna(row['latitude']) or row['latitude'] == 0
        lng_missing = pd.isna(row['longitude']) or row['longitude'] == 0
        
        if (lat_missing or lng_missing) and not pd.isna(row.get('location_mentioned')):
            location = row['location_mentioned']
            
            # Try to geocode the location
            print(f"  Geocoding: {location}")
            lat, lng = geocode_location(location)
            
            # If first method fails, try alternative
            if lat is None and lng is None:
                lat, lng = get_coordinates_from_nominatim(location)
            
            if lat is not None and lng is not None:
                enhanced_df.at[idx, 'latitude'] = lat
                enhanced_df.at[idx, 'longitude'] = lng
                enhanced_count += 1
                print(f"    → Found coordinates: ({lat:.4f}, {lng:.4f})")
            else:
                print(f"    → Could not geocode location")
    
    print(f"Enhanced {enhanced_count} records with coordinates from location_mentioned")
    return enhanced_df

def clean_data(df):
    """
    Clean the data by removing rows with missing coordinates after enhancement
    """
    original_count = len(df)
    
    # First enhance data with coordinates from location_mentioned
    df_enhanced = enhance_data_with_coordinates(df)
    
    # Remove rows where latitude or longitude is still NaN
    df_clean = df_enhanced.dropna(subset=['latitude', 'longitude'])
    
    # Also remove rows where coordinates are 0,0 (common placeholder for missing data)
    df_clean = df_clean[(df_clean['latitude'] != 0) & (df_clean['longitude'] != 0)]
    
    cleaned_count = len(df_clean)
    removed_count = original_count - cleaned_count
    
    if removed_count > 0:
        print(f"Removed {removed_count} records with missing/invalid coordinates")
        print(f"Working with {cleaned_count} valid records")
    
    return df_clean

def create_disaster_heatmap(output_file='disaster_heatmap.html'):
    """
    Create a geographic heatmap from disaster data CSV with interactive markers
    """
    # Read and clean the data from CSV
    df = pd.read_csv(CSV_FILE_PATH)
    df = clean_data(df)
    
    if len(df) == 0:
        raise ValueError("No valid records with coordinates found after cleaning")
    
    # Check required columns
    required_columns = ['latitude', 'longitude', 'severity_level', 'model_confidence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert severity to numerical weights for heatmap intensity
    severity_weights = {
        'high': 1.0,
        'medium': 0.7, 
        'low': 0.4
    }
    
    # Create heatmap data: [lat, lng, intensity]
    # Intensity = severity_weight * confidence_score
    heatmap_data = []
    for _, row in df.iterrows():
        lat = row['latitude']
        lng = row['longitude']
        severity_weight = severity_weights.get(str(row['severity_level']).lower(), 0.5)
        confidence = row['model_confidence']
        
        # Calculate intensity (0-1 scale)
        intensity = severity_weight * confidence
        heatmap_data.append([lat, lng, intensity])
    
    # Calculate map center from data
    center_lat = df['latitude'].mean()
    center_lng = df['longitude'].mean()
    
    print(f"Map center: ({center_lat:.4f}, {center_lng:.4f})")
    print(f"Heatmap data points: {len(heatmap_data)}")
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add heatmap layer
    HeatMap(
        heatmap_data,
        name='Disaster Intensity Heatmap',
        min_opacity=0.3,
        max_opacity=0.8,
        radius=25,  # Increased radius for smoother heatmap
        blur=20,    # Increased blur for smoother transitions
        gradient={
            0.0: 'blue',      # Low intensity
            0.3: 'cyan',      # Medium-low
            0.5: 'lime',      # Medium
            0.7: 'yellow',    # Medium-high
            0.9: 'red'        # High intensity
        }
    ).add_to(m)
    
    # ADD INDIVIDUAL MARKERS WITH POPUPS TO THE MAIN HEATMAP
    markers_group = folium.FeatureGroup(name='Individual Events', show=True)
    
    # Color mapping for disaster types
    disaster_colors = {
        'fire': 'red',
        'shooting': 'black', 
        'flood': 'blue',
        'hurricane': 'cyan',
        'earthquake': 'orange',
        'auto_accident': 'purple',
        'severe_storm': 'gray',
        'other_disaster': 'pink',
        'not_relevant': 'white',
        'extreme_heat': 'darkred'
    }
    
    for _, row in df.iterrows():
        disaster_type = row.get('disaster_type', 'other_disaster')
        color = disaster_colors.get(disaster_type, 'green')
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,  # Slightly larger for better visibility
            popup=folium.Popup(
                f"""
                <div style="width: 250px;">
                    <h4>Disaster Event Details</h4>
                    <b>Type:</b> {disaster_type}<br>
                    <b>Severity:</b> {row['severity_level']}<br>
                    <b>Confidence:</b> {row['model_confidence']:.3f}<br>
                    <b>Location:</b> {row.get('location_mentioned', 'N/A')}<br>
                    <b>Help Request:</b> {row.get('help_request', 'N/A')}<br>
                    <b>Intensity Score:</b> {severity_weights.get(str(row['severity_level']).lower(), 0.5) * row['model_confidence']:.3f}
                </div>
                """,
                max_width=300
            ),
            tooltip=f"{disaster_type} - {row['severity_level']} severity",
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=2
        ).add_to(markers_group)
    
    markers_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:20px"><b>Disaster Event Intensity Heatmap</b></h3>
    <p align="center">Color intensity represents combined severity and confidence</p>
    <p align="center">Click on colored circles to see individual event details</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 280px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Heatmap Intensity Legend</b></p>
    <p><span style="background: linear-gradient(to right, blue, cyan, lime, yellow, red); 
                    display: block; height: 20px; width: 100%;"></span></p>
    <p>Low ← Intensity → High</p>
    <p>Intensity = Severity × Confidence</p>
    <p><b>Marker Colors:</b><br>
    🔴 Fire ⚫ Shooting 🔵 Flood<br>
    🟠 Earthquake 🟣 Auto Accident</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(output_file)
    print(f"Heatmap saved as: {output_file}")
    print("→ Both heatmap AND markers are visible by default")
    print("→ Click on colored circles to see popups")
    print("→ Use layer control (top-right) to toggle layers")
    
    return m, heatmap_data

def create_simple_heatmap_with_markers(output_file='disaster_heatmap_simple.html'):
    """
    Create a hybrid heatmap that also shows individual disaster markers
    """
    df = pd.read_csv(CSV_FILE_PATH)
    df = clean_data(df)
    
    if len(df) == 0:
        raise ValueError("No valid records with coordinates found after cleaning")
    
    # Calculate center
    center_lat = df['latitude'].mean()
    center_lng = df['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=5
    )
    
    # Prepare heatmap data (simpler version)
    heat_data = [[row['latitude'], row['longitude'], row['model_confidence']] 
                 for _, row in df.iterrows()]
    
    # Add heatmap
    HeatMap(
        heat_data,
        name='Disaster Confidence Heatmap',
        min_opacity=0.4,
        max_opacity=0.8,
        radius=20,
        blur=15,
        gradient={'0.0': 'blue', '0.5': 'lime', '1.0': 'red'}
    ).add_to(m)
    
    # Add individual markers with popups - VISIBLE BY DEFAULT
    feature_group = folium.FeatureGroup(name='Individual Events', show=True)
    
    # Color mapping for disaster types
    disaster_colors = {
        'fire': 'red',
        'shooting': 'black', 
        'flood': 'blue',
        'hurricane': 'cyan',
        'earthquake': 'orange',
        'auto_accident': 'purple',
        'severe_storm': 'gray',
        'other_disaster': 'pink',
        'not_relevant': 'white',
        'extreme_heat': 'darkred'
    }
    
    for _, row in df.iterrows():
        disaster_type = row.get('disaster_type', 'other_disaster')
        color = disaster_colors.get(disaster_type, 'green')
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,  # Slightly larger for better visibility
            popup=folium.Popup(
                f"""
                <div style="width: 250px;">
                    <h4>Disaster Event</h4>
                    <b>Type:</b> {disaster_type}<br>
                    <b>Severity:</b> {row['severity_level']}<br>
                    <b>Confidence:</b> {row['model_confidence']:.3f}<br>
                    <b>Location:</b> {row.get('location_mentioned', 'N/A')}<br>
                    <b>Help Request:</b> {row.get('help_request', 'N/A')}
                </div>
                """,
                max_width=300
            ),
            tooltip=f"{disaster_type} - {row['severity_level']} severity",
            color=color,
            fill=True,
            fillOpacity=0.8
        ).add_to(feature_group)
    
    feature_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Add title and legend
    title_html = '''
    <h3 align="center" style="font-size:20px"><b>Disaster Events - Heatmap & Markers</b></h3>
    <p align="center">Click on colored circles to see event details</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(output_file)
    print(f"Hybrid heatmap saved as: {output_file}")
    print("→ Markers are VISIBLE by default")
    print("→ Click on colored circles to see popups")
    return m

def print_data_summary():
    """
    Print a summary of the data for verification
    """
    df = pd.read_csv(CSV_FILE_PATH)
    
    print("\n=== DATA SUMMARY ===")
    print(f"Total records: {len(df)}")
    
    # Check for missing coordinates before enhancement
    missing_lat = df['latitude'].isna().sum()
    missing_lng = df['longitude'].isna().sum()
    zero_coords = ((df['latitude'] == 0) & (df['longitude'] == 0)).sum()
    missing_location = df['location_mentioned'].isna().sum()
    
    print(f"Records with missing latitude: {missing_lat}")
    print(f"Records with missing longitude: {missing_lng}")
    print(f"Records with (0,0) coordinates: {zero_coords}")
    print(f"Records with missing location_mentioned: {missing_location}")
    
    # Count records that can potentially be enhanced
    enhanceable_records = 0
    for _, row in df.iterrows():
        lat_missing = pd.isna(row['latitude']) or row['latitude'] == 0
        lng_missing = pd.isna(row['longitude']) or row['longitude'] == 0
        has_location = not pd.isna(row.get('location_mentioned'))
        
        if (lat_missing or lng_missing) and has_location:
            enhanceable_records += 1
    
    print(f"Records that can be enhanced with location_mentioned: {enhanceable_records}")
    
    # Clean data for the rest of the summary
    df_clean = clean_data(df)
    
    print(f"Valid records with coordinates: {len(df_clean)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Disaster types: {df['disaster_type'].unique()}")
    print(f"Severity levels: {df['severity_level'].unique()}")
    print(f"Confidence range: {df['model_confidence'].min():.3f} to {df['model_confidence'].max():.3f}")
    
    if len(df_clean) > 0:
        print(f"Latitude range: {df_clean['latitude'].min():.3f} to {df_clean['latitude'].max():.3f}")
        print(f"Longitude range: {df_clean['longitude'].min():.3f} to {df_clean['longitude'].max():.3f}")
    print("==================\n")

# Install required packages if not already installed
def check_dependencies():
    try:
        from geopy.geocoders import Nominatim
        import requests
        return True
    except ImportError:
        print("Required packages missing. Please install them:")
        print("pip install geopy requests")
        return False

# Usage example:
if __name__ == "__main__":
    if not check_dependencies():
        exit(1)
        
    try:
        # Print data summary first
        print_data_summary()
        
        # Create the main heatmap (now with markers!)
        heatmap, heatmap_data = create_disaster_heatmap()
        
        # Create hybrid version with markers
        hybrid_map = create_simple_heatmap_with_markers()
        
        print("Heatmaps created successfully!")
        print(f"Processed {len(heatmap_data)} disaster events from: {CSV_FILE_PATH}")
        print("Generated files:")
        print("  - disaster_heatmap.html (heatmap + markers + enhanced legend)")
        print("  - disaster_heatmap_simple.html (heatmap + markers)")
        
    except FileNotFoundError:
        print(f"CSV file not found: {CSV_FILE_PATH}")
        print("Please update the CSV_FILE_PATH variable at the top of the script")
    except Exception as e:
        print(f"Error creating heatmap: {e}")