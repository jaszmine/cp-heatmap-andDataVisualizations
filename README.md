 𝜗𝜚 ࣪˖ ִ𐙚 heyyyyy divasss ✧˖°⊹ ࣪ ˖

## visualize.py
**Purpose**: provides comprehensive data visualization and analysis for disaster-related social media posts, generating insightful charts, metrics, and interactive maps from CSV data.

**Features**:
- Data Preprocessing: Cleans and prepares disaster data with timestamp conversion and location handling
- Multiple Visualization Types: Creates pie charts, bar charts, line graphs, and box plots
- Interactive Geographic Mapping: Generates Folium maps with disaster location markers
- Temporal Analysis: Shows daily trends and hourly patterns of disaster reports
- Severity Assessment: Analyzes disaster severity levels with model confidence metrics
- Help Request Tracking: Identifies posts requesting assistance by disaster type

**Outputs**: (generates several visualization files as pngs)
- `disaster_type_distribution.png` - Disaster frequency by type
- `severity_analysis.png` - Severity levels and confidence scores
- `temporal_analysis.png` - Daily and hourly posting patterns
- `help_request_analysis.png` - Help requests across disaster types
- `model_confidence_analysis.png` - AI model performance by disaster type
- `location_analysis.png` - Top mentioned locations
- `author_analysis.png` - Most active reporters
- `disaster_geographic_map.html` - Interactive map with disaster locations

**Usage**:
```bash
# Initialize with your CSV file
visualizer = DisasterDataVisualizer("your_disaster_data.csv")

# Run complete analysis
visualizer.run_all_analyses()
# -- OR -->
python3 visualize.py # ensure you're in src on terminal

# Or run individual analyses
visualizer.disaster_type_distribution()
visualizer.create_geographic_map()
visualizer.generate_dashboard_metrics()
```

**Data Requirements**:
The CSV file should contain columns for:
- `disaster_type` (categorical)
- `severity_level` (high/medium/low)
- `location_mentioned` (text)
- `latitude/longitude` (coordinates)
- `help_request` (boolean)
- `model_confidence` (0-1 score)
- `author` (text)
- `processed_at` (timestamp)

**Dependencies**:
- pandas, matplotlib, seaborn - Data analysis and static visualizations
- folium - Interactive geographic maps
- plotly - Advanced interactive charts
- numpy - Numerical computations

## heatmap.py
**Purpose**: creates interactive geographic heatmaps from disaster data, automatically enhancing location coordinates and visualizing disaster intensity across regions.

**Features**:
- `Automatic Geocoding`: Converts location names to coordinates using OpenStreetMap's Nominatim service
- `Data Enhancement`: Fills in missing coordinates using location mentions
- `Interactive Heatmaps`: Creates layered Folium maps with both heatmap overlays and individual markers
- `Intensity Calculation`: Combines severity levels and model confidence for dynamic heatmap weighting
- `Dual Output`: Generates both comprehensive and simplified map versions

**Outputs**:
The script generates two interactive HTML maps:
- `disaster_heatmap.html` - Comprehensive version with:
   - Color-coded intensity heatmap (blue → red gradient)
   - Individual disaster markers with detailed popups
   - Interactive layer controls
   - Built-in legend and instructions
   - Severity-weighted intensity calculation
- `disaster_heatmap_simple.html` - Streamlined version with:
   - Confidence-based heatmap
   - Visible disaster markers by default
   - Clean, focused visualization

**How it Works**:
1. Data Loading & Cleaning:
   - Reads CSV disaster data
   - Identifies records with missing coordinates
   - Uses geocoding to enhance incomplete data
2. Coordinate Enhancement:
   - Attempts multiple geocoding methods (geopy + direct API)
   - Adds delays to respect rate limits
   - Provides detailed progress reporting
3. Intensity Calculation:
   - Maps severity levels to weights (High: 1.0, Medium: 0.7, Low: 0.4)
   - Calculates intensity = severity weight × confidence score
   - Creates gradient from low (blue) to high (red) intensity
4. Map Generation:
   - Automatic center calculation from data
   - Color-coded markers by disaster type
   - Detailed popups with event information
   - Layer controls to toggle heatmap/markers

**Usage**:
```bash
# Basic usage (run the script directly)
python3 disaster_heatmap.py

# Or use the functions individually:
from disaster_heatmap import create_disaster_heatmap, print_data_summary

# Check your data first
print_data_summary()

# Generate the main heatmap
heatmap, data = create_disaster_heatmap('custom_output.html')

# Generate simplified version
simple_map = create_simple_heatmap_with_markers('simple_map.html')
```

**Data Requirements**: Same as visualize.py

**Dependencies**:
- folium - Interactive map generation
- geopy - Location geocoding services
- pandas - Data processing and cleaning
- requests - API calls for geocoding
- matplotlib/seaborn - Optional data analysis

**Features for Emergency Response**:
- Rapid Assessment: Visualize disaster concentration areas
- Severity Prioritization: Identify high-intensity zones
- Resource Allocation: See help request patterns
- Multi-layer Analysis: Toggle between overview and individual events
