import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from datetime import datetime
import numpy as np

# Set up styling
plt.style.use('default')
sns.set_palette("husl")

class DisasterDataVisualizer:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        # self.df = pd.read_csv("114records-be_extracted_info_output_rows.csv")
        self.df = pd.read_csv("2006records-be_extracted_info_output_rows.csv")
        self.preprocess_data()
        
    # def preprocess_data(self):
    #     """Clean and preprocess the data"""
    #     # Convert timestamp columns
    #     self.df['processed_at'] = pd.to_datetime(self.df['processed_at'])
    #     self.df['indexed_at'] = pd.to_datetime(self.df['indexed_at'])
        
    #     # Extract time components
    #     self.df['hour'] = self.df['processed_at'].dt.hour
    #     self.df['date'] = self.df['processed_at'].dt.date
    #     self.df['day_of_week'] = self.df['processed_at'].dt.day_name()
        
    #     # Clean location data
    #     self.df['location_mentioned'] = self.df['location_mentioned'].replace('null', None)
        
    #     print(f"Loaded {len(self.df)} disaster posts")
    #     print(f"Date range: {self.df['processed_at'].min()} to {self.df['processed_at'].max()}")
    """ fixed - timestamp parsing now accounts for timezone offset (+00)  """
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Convert timestamp columns - handle timezone format
        try:
            self.df['processed_at'] = pd.to_datetime(self.df['processed_at'], format='ISO8601')
            self.df['indexed_at'] = pd.to_datetime(self.df['indexed_at'], format='ISO8601')
        except:
            # Fallback to mixed format parsing
            self.df['processed_at'] = pd.to_datetime(self.df['processed_at'], format='mixed')
            self.df['indexed_at'] = pd.to_datetime(self.df['indexed_at'], format='mixed')
        
        # Extract time components
        self.df['hour'] = self.df['processed_at'].dt.hour
        self.df['date'] = self.df['processed_at'].dt.date
        self.df['day_of_week'] = self.df['processed_at'].dt.day_name()
        
        # Clean location data
        self.df['location_mentioned'] = self.df['location_mentioned'].replace('null', None)
        
        print(f"Loaded {len(self.df)} disaster posts")
        print(f"Date range: {self.df['processed_at'].min()} to {self.df['processed_at'].max()}")
    
    # def disaster_type_distribution(self):
    #     """Disaster type distribution pie chart and bar chart"""
    #     disaster_counts = self.df['disaster_type'].value_counts()
        
    #     # Create subplots
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
    #     # Pie chart
    #     ax1.pie(disaster_counts.values, labels=disaster_counts.index, autopct='%1.1f%%', startangle=90)
    #     ax1.set_title('Disaster Type Distribution', fontsize=14, fontweight='bold')
        
    #     # Bar chart
    #     bars = ax2.bar(disaster_counts.index, disaster_counts.values, color='skyblue')
    #     ax2.set_title('Disaster Type Counts', fontsize=14, fontweight='bold')
    #     ax2.set_xlabel('Disaster Type')
    #     ax2.set_ylabel('Number of Posts')
    #     ax2.tick_params(axis='x', rotation=45)
        
    #     # Add value labels on bars
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax2.text(bar.get_x() + bar.get_width()/2., height,
    #                 f'{int(height)}', ha='center', va='bottom')
        
    #     plt.tight_layout()
    #     plt.savefig('disaster_type_distribution.png', dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     return disaster_counts
    """ fixed - using legend for pize chart (so that text doesn't overlap for smaller sections) """
    """ also added interactice localHost feature """
    def disaster_type_distribution(self):
        """Interactive disaster type distribution using Plotly"""
        disaster_counts = self.df['disaster_type'].value_counts()
        
        # Create interactive pie chart
        fig = make_subplots(rows=1, cols=2, 
                        specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=('Disaster Type Distribution', 'Disaster Type Counts'))
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=disaster_counts.index, 
                values=disaster_counts.values,
                textinfo='percent+label',
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=disaster_counts.index, 
                y=disaster_counts.values,
                marker_color='lightblue',
                text=disaster_counts.values,
                textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Disaster Type Analysis",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        fig.write_html('disaster_type_interactive.html')
        fig.show()
        
        return disaster_counts
    
    def severity_analysis(self):
        """Severity level analysis with confidence"""
        severity_data = self.df.groupby('severity_level').agg({
            'uri': 'count',
            'model_confidence': 'mean'
        }).rename(columns={'uri': 'post_count', 'model_confidence': 'avg_confidence'})
        
        # Order by severity
        severity_order = ['high', 'medium', 'low']
        severity_data = severity_data.reindex(severity_order)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Post count by severity
        bars = ax1.bar(severity_data.index, severity_data['post_count'], color=['red', 'orange', 'green'])
        ax1.set_title('Posts by Severity Level', fontweight='bold')
        ax1.set_ylabel('Number of Posts')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Confidence by severity
        bars2 = ax2.bar(severity_data.index, severity_data['avg_confidence'], color=['red', 'orange', 'green'])
        ax2.set_title('Average Model Confidence by Severity', fontweight='bold')
        ax2.set_ylabel('Average Confidence')
        ax2.set_ylim(0, 1)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('severity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return severity_data
    
    def temporal_analysis(self):
        """Posts over time analysis"""
        # Daily trend
        daily_posts = self.df.groupby(self.df['processed_at'].dt.date).size()
        
        # Hourly pattern
        hourly_posts = self.df.groupby('hour').size()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Daily trend
        ax1.plot(daily_posts.index, daily_posts.values, marker='o', linewidth=2, markersize=4)
        ax1.set_title('Daily Disaster Posts Trend', fontweight='bold')
        ax1.set_ylabel('Number of Posts')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Hourly pattern
        ax2.bar(hourly_posts.index, hourly_posts.values, color='purple', alpha=0.7)
        ax2.set_title('Hourly Distribution of Disaster Posts', fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Posts')
        ax2.set_xticks(range(0, 24))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return daily_posts, hourly_posts
    
    def create_geographic_map(self):
        """Create an interactive geographic heat map"""
        # Filter posts with coordinates
        geo_df = self.df.dropna(subset=['latitude', 'longitude'])
        
        if len(geo_df) == 0:
            print("No geographic data available for mapping")
            return None
        
        # Create base map
        center_lat = geo_df['latitude'].mean()
        center_lon = geo_df['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        # Color mapping for disaster types
        disaster_colors = {
            'fire': 'red',
            'earthquake': 'orange', 
            'flood': 'blue',
            'auto_accident': 'purple',
            'severe_storm': 'gray',
            'shooting': 'black',
            'tornado': 'brown',
            'hurricane': 'cyan',
            'other_disaster': 'pink'
        }
        
        # Add markers for each post
        for idx, row in geo_df.iterrows():
            color = disaster_colors.get(row['disaster_type'], 'green')
            
            popup_text = f"""
            <b>Disaster:</b> {row['disaster_type']}<br>
            <b>Severity:</b> {row['severity_level']}<br>
            <b>Location:</b> {row['location_mentioned'] or 'Not specified'}<br>
            <b>Confidence:</b> {row['model_confidence']:.2f}<br>
            <b>Help Request:</b> {row['help_request']}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
        
        # Save map
        m.save('disaster_geographic_map.html')
        print("Interactive map saved as 'disaster_geographic_map.html'")
        
        return m
    
    def help_request_analysis(self):
        """Analyze help requests by disaster type"""
        help_analysis = self.df.groupby(['disaster_type', 'help_request']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if True in help_analysis.columns and False in help_analysis.columns:
            help_analysis.plot(kind='bar', ax=ax, color=['lightcoral', 'lightblue'])
        elif True in help_analysis.columns:
            help_analysis.plot(kind='bar', ax=ax, color='lightcoral')
        else:
            help_analysis.plot(kind='bar', ax=ax, color='lightblue')
            
        ax.set_title('Help Requests by Disaster Type', fontweight='bold')
        ax.set_ylabel('Number of Posts')
        ax.set_xlabel('Disaster Type')
        ax.legend(['Help Requested', 'No Help Requested'])
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('help_request_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return help_analysis
    
    def model_confidence_analysis(self):
        """Analyze model confidence across disaster types"""
        confidence_stats = self.df.groupby('disaster_type').agg({
            'model_confidence': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        
        confidence_stats.columns = ['count', 'mean', 'std', 'min', 'max']
        confidence_stats = confidence_stats.sort_values('mean', ascending=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Better approach with seaborn
        import seaborn as sns
        sns.boxplot(data=self.df, 
                    y='disaster_type', 
                    x='model_confidence',
                    palette='viridis',  # Adds colors
                    ax=ax)
        
        ax.set_title('Model Confidence Distribution by Disaster Type', fontweight='bold')
        ax.set_ylabel('Disaster Type')
        ax.set_xlabel('Confidence Score')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig('model_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return confidence_stats
    # def model_confidence_analysis(self):
    #     """Analyze model confidence across disaster types"""
    #     confidence_stats = self.df.groupby('disaster_type').agg({
    #         'model_confidence': ['count', 'mean', 'std', 'min', 'max']
    #     }).round(3)
        
    #     confidence_stats.columns = ['count', 'mean', 'std', 'min', 'max']
    #     confidence_stats = confidence_stats.sort_values('mean', ascending=False)
        
    #     # Create visualization
    #     fig, ax = plt.subplots(figsize=(12, 6))
        
    #     # Box plot of confidence by disaster type
    #     self.df.boxplot(column='model_confidence', by='disaster_type', ax=ax)
    #     ax.set_title('Model Confidence Distribution by Disaster Type', fontweight='bold')
    #     ax.set_ylabel('Confidence Score')
    #     ax.set_xlabel('Disaster Type')
    #     plt.suptitle('')  # Remove automatic title
        
    #     plt.tight_layout()
    #     plt.savefig('model_confidence_analysis.png', dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     return confidence_stats
    
    def location_analysis(self):
        """Analyze most mentioned locations"""
        location_counts = self.df['location_mentioned'].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(location_counts.index, location_counts.values, color='teal')
        ax.set_title('Top 15 Most Mentioned Locations', fontweight='bold')
        ax.set_xlabel('Number of Mentions')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., 
                   f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('location_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return location_counts
    
    # def author_analysis(self):
    #     """Analyze authors with multiple posts"""
    #     author_counts = self.df['author'].value_counts()
    #     multiple_authors = author_counts[author_counts > 1]
        
    #     if len(multiple_authors) > 0:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         bars = ax.bar(multiple_authors.index, multiple_authors.values, color='orange')
    #         ax.set_title('Authors with Multiple Disaster Reports', fontweight='bold')
    #         ax.set_ylabel('Number of Posts')
    #         ax.tick_params(axis='x', rotation=45)
            
    #         for bar in bars:
    #             height = bar.get_height()
    #             ax.text(bar.get_x() + bar.get_width()/2., height,
    #                    f'{int(height)}', ha='center', va='bottom')
            
    #         plt.tight_layout()
    #         plt.savefig('author_analysis.png', dpi=300, bbox_inches='tight')
    #         plt.show()
        
    #     return multiple_authors
    def author_analysis(self):
        """Analyze authors with 10 or more classified posts"""
        author_counts = self.df['author'].value_counts()
        
        # Filter for authors with 10+ posts
        frequent_authors = author_counts[author_counts >= 10]
        
        if len(frequent_authors) > 0:
            print(f"Found {len(frequent_authors)} authors with 10+ posts")
            
            # Use horizontal layout for better readability
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(frequent_authors.index, frequent_authors.values, color='orange')
            ax.set_title('Authors with 10+ Disaster Posts', fontweight='bold', fontsize=14)
            ax.set_xlabel('Number of Posts', fontweight='bold')
            ax.set_ylabel('Author', fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2., 
                    f'{int(width)}', ha='left', va='center', 
                    fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('author_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            print("No authors found with 10 or more posts")
            # Optional: Show authors with 5+ posts instead
            alternative_authors = author_counts[author_counts >= 5]
            if len(alternative_authors) > 0:
                print(f"Showing authors with 5+ posts instead: {len(alternative_authors)} authors")
                # You could create a chart for these as well
        
        return frequent_authors
    
    def generate_dashboard_metrics(self):
        """Generate key metrics for dashboard"""
        metrics = {
            'total_disaster_posts': len(self.df),
            'unique_disaster_types': self.df['disaster_type'].nunique(),
            'unique_locations': self.df['location_mentioned'].nunique(),
            'overall_confidence': self.df['model_confidence'].mean(),
            'help_requests_count': self.df['help_request'].sum(),
            'latest_post_time': self.df['processed_at'].max(),
            'most_common_disaster': self.df['disaster_type'].mode().iloc[0] if not self.df.empty else 'N/A'
        }
        
        print("=== DASHBOARD METRICS ===")
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return metrics
    
    def run_all_analyses(self):
        """Run all visualizations and analyses"""
        print("Starting comprehensive disaster data analysis...")
        
        # Generate all visualizations
        self.generate_dashboard_metrics()
        self.disaster_type_distribution()
        self.severity_analysis()
        self.temporal_analysis()
        self.help_request_analysis()
        self.model_confidence_analysis()
        self.location_analysis()
        self.author_analysis()
        
        # Create interactive map
        self.create_geographic_map()
        
        print("\nAll analyses completed! Check the generated files:")
        print("- PNG images for charts")
        print("- HTML file for interactive map")

if __name__ == "__main__":
    csv_file = "be_extracted_info_output.csv"
    
    try:
        visualizer = DisasterDataVisualizer(csv_file)
        visualizer.run_all_analyses()
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("Please update the csv_file path in the script.")
    except Exception as e:
        print(f"Error during analysis: {e}")