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

