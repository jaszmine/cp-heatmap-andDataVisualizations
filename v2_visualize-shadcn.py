import pandas as pd
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Any

class DisasterDataVisualizer:
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.df = pd.read_csv("2006records-be_extracted_info_output_rows.csv")
        self.preprocess_data()
        self.chart_data = {}
        
    def preprocess_data(self):
        """Clean and preprocess the data"""
        try:
            self.df['processed_at'] = pd.to_datetime(self.df['processed_at'], format='ISO8601')
            self.df['indexed_at'] = pd.to_datetime(self.df['indexed_at'], format='ISO8601')
        except:
            self.df['processed_at'] = pd.to_datetime(self.df['processed_at'], format='mixed')
            self.df['indexed_at'] = pd.to_datetime(self.df['indexed_at'], format='mixed')
        
        self.df['hour'] = self.df['processed_at'].dt.hour
        self.df['date'] = self.df['processed_at'].dt.date
        self.df['day_of_week'] = self.df['processed_at'].dt.day_name()
        self.df['location_mentioned'] = self.df['location_mentioned'].replace('null', None)
        
        print(f"Loaded {len(self.df)} disaster posts")
        print(f"Date range: {self.df['processed_at'].min()} to {self.df['processed_at'].max()}")

    def disaster_type_distribution(self):
        """Disaster type distribution for shadcn charts"""
        disaster_counts = self.df['disaster_type'].value_counts()
        
        # Pie chart data
        pie_data = {
            "type": "pie",
            "data": {
                "labels": disaster_counts.index.tolist(),
                "datasets": [{
                    "data": disaster_counts.values.tolist(),
                    "backgroundColor": [
                        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
                        "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43"
                    ]
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Disaster Type Distribution"
                    },
                    "legend": {
                        "position": "right"
                    }
                }
            }
        }
        
        # Bar chart data
        bar_data = {
            "type": "bar",
            "data": {
                "labels": disaster_counts.index.tolist(),
                "datasets": [{
                    "label": "Number of Posts",
                    "data": disaster_counts.values.tolist(),
                    "backgroundColor": "#3b82f6",
                    "borderColor": "#1d4ed8",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Disaster Type Counts"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        }
        
        self.chart_data['disaster_type_pie'] = pie_data
        self.chart_data['disaster_type_bar'] = bar_data
        
        return disaster_counts

    def severity_analysis(self):
        """Severity level analysis for shadcn charts"""
        severity_data = self.df.groupby('severity_level').agg({
            'uri': 'count',
            'model_confidence': 'mean'
        }).rename(columns={'uri': 'post_count', 'model_confidence': 'avg_confidence'})
        
        severity_order = ['high', 'medium', 'low']
        severity_data = severity_data.reindex(severity_order)
        
        # Posts by severity
        posts_chart = {
            "type": "bar",
            "data": {
                "labels": severity_data.index.tolist(),
                "datasets": [{
                    "label": "Number of Posts",
                    "data": severity_data['post_count'].tolist(),
                    "backgroundColor": ["#ef4444", "#f59e0b", "#10b981"],
                    "borderColor": ["#dc2626", "#d97706", "#059669"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Posts by Severity Level"
                    }
                }
            }
        }
        
        # Confidence by severity
        confidence_chart = {
            "type": "bar",
            "data": {
                "labels": severity_data.index.tolist(),
                "datasets": [{
                    "label": "Average Confidence",
                    "data": severity_data['avg_confidence'].round(3).tolist(),
                    "backgroundColor": ["#ef4444", "#f59e0b", "#10b981"],
                    "borderColor": ["#dc2626", "#d97706", "#059669"],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Average Model Confidence by Severity"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0
                    }
                }
            }
        }
        
        self.chart_data['severity_posts'] = posts_chart
        self.chart_data['severity_confidence'] = confidence_chart
        
        return severity_data

    def temporal_analysis(self):
        """Temporal analysis for shadcn charts"""
        # Daily trend
        daily_posts = self.df.groupby(self.df['processed_at'].dt.date).size()
        
        # Hourly pattern
        hourly_posts = self.df.groupby('hour').size()
        
        # Daily trend chart
        daily_chart = {
            "type": "line",
            "data": {
                "labels": [str(date) for date in daily_posts.index],
                "datasets": [{
                    "label": "Daily Posts",
                    "data": daily_posts.values.tolist(),
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "tension": 0.4,
                    "fill": True
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Daily Disaster Posts Trend"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Date"
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Number of Posts"
                        },
                        "beginAtZero": True
                    }
                }
            }
        }
        
        # Hourly pattern chart
        hourly_chart = {
            "type": "bar",
            "data": {
                "labels": [f"{h:02d}:00" for h in range(24)],
                "datasets": [{
                    "label": "Posts per Hour",
                    "data": [hourly_posts.get(h, 0) for h in range(24)],
                    "backgroundColor": "#8b5cf6",
                    "borderColor": "#7c3aed",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Hourly Distribution of Disaster Posts"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Hour of Day"
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Number of Posts"
                        },
                        "beginAtZero": True
                    }
                }
            }
        }
        
        self.chart_data['temporal_daily'] = daily_chart
        self.chart_data['temporal_hourly'] = hourly_chart
        
        return daily_posts, hourly_posts

    def help_request_analysis(self):
        """Help request analysis for shadcn charts"""
        help_analysis = self.df.groupby(['disaster_type', 'help_request']).size().unstack(fill_value=0)
        
        labels = help_analysis.index.tolist()
        help_requested = help_analysis.get(True, pd.Series(0, index=labels)).tolist()
        no_help = help_analysis.get(False, pd.Series(0, index=labels)).tolist()
        
        chart_data = {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Help Requested",
                        "data": help_requested,
                        "backgroundColor": "#ef4444"
                    },
                    {
                        "label": "No Help Requested",
                        "data": no_help,
                        "backgroundColor": "#d1d5db"
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Help Requests by Disaster Type"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Disaster Type"
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Number of Posts"
                        },
                        "beginAtZero": True
                    }
                }
            }
        }
        
        self.chart_data['help_requests'] = chart_data
        return help_analysis

    def model_confidence_analysis(self):
        """Model confidence analysis for shadcn charts"""
        confidence_stats = self.df.groupby('disaster_type').agg({
            'model_confidence': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        
        confidence_stats.columns = ['count', 'mean', 'std', 'min', 'max']
        confidence_stats = confidence_stats.sort_values('mean', ascending=False)
        
        # Box plot equivalent using scatter with error bars
        disaster_types = confidence_stats.index.tolist()
        means = confidence_stats['mean'].tolist()
        stds = confidence_stats['std'].fillna(0).tolist()
        
        chart_data = {
            "type": "scatter",
            "data": {
                "labels": disaster_types,
                "datasets": [{
                    "label": "Average Confidence",
                    "data": [{"x": dt, "y": mean, "error": std} for dt, mean, std in zip(disaster_types, means, stds)],
                    "backgroundColor": "#10b981",
                    "borderColor": "#059669",
                    "pointRadius": 8,
                    "pointHoverRadius": 10
                }]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Model Confidence by Disaster Type"
                    },
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { return 'Confidence: ' + context.parsed.y.toFixed(3) + ' ± ' + context.raw.error.toFixed(3); }"
                        }
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Disaster Type"
                        }
                    },
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Confidence Score"
                        },
                        "min": 0,
                        "max": 1
                    }
                }
            }
        }
        
        self.chart_data['confidence_analysis'] = chart_data
        return confidence_stats

    def location_analysis(self):
        """Location analysis for shadcn charts"""
        location_counts = self.df['location_mentioned'].value_counts().head(15)
        
        chart_data = {
            "type": "bar",
            "data": {
                "labels": location_counts.index.tolist(),
                "datasets": [{
                    "label": "Number of Mentions",
                    "data": location_counts.values.tolist(),
                    "backgroundColor": "#06b6d4",
                    "borderColor": "#0891b2",
                    "borderWidth": 1
                }]
            },
            "options": {
                "indexAxis": "y",
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Top 15 Most Mentioned Locations"
                    }
                },
                "scales": {
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Number of Mentions"
                        },
                        "beginAtZero": True
                    }
                }
            }
        }
        
        self.chart_data['location_analysis'] = chart_data
        return location_counts

    def author_analysis(self):
        """Author analysis for shadcn charts"""
        author_counts = self.df['author'].value_counts()
        frequent_authors = author_counts[author_counts >= 10]
        
        if len(frequent_authors) > 0:
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": frequent_authors.index.tolist(),
                    "datasets": [{
                        "label": "Number of Posts",
                        "data": frequent_authors.values.tolist(),
                        "backgroundColor": "#f59e0b",
                        "borderColor": "#d97706",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "indexAxis": "y",
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Authors with 10+ Disaster Posts"
                        }
                    },
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Number of Posts"
                            },
                            "beginAtZero": True
                        }
                    }
                }
            }
            
            self.chart_data['author_analysis'] = chart_data
        
        return frequent_authors

    def generate_dashboard_metrics(self):
        """Generate key metrics for dashboard"""
        metrics = {
            'total_disaster_posts': len(self.df),
            'unique_disaster_types': self.df['disaster_type'].nunique(),
            'unique_locations': self.df['location_mentioned'].nunique(),
            'overall_confidence': round(self.df['model_confidence'].mean(), 3),
            'help_requests_count': int(self.df['help_request'].sum()),
            'latest_post_time': self.df['processed_at'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'most_common_disaster': self.df['disaster_type'].mode().iloc[0] if not self.df.empty else 'N/A',
            'high_severity_count': len(self.df[self.df['severity_level'] == 'high']),
            'data_time_range': f"{self.df['processed_at'].min().strftime('%Y-%m-%d')} to {self.df['processed_at'].max().strftime('%Y-%m-%d')}"
        }
        
        self.chart_data['metrics'] = metrics
        return metrics

    def get_geographic_data(self):
        """Get geographic data for mapping components"""
        geo_df = self.df.dropna(subset=['latitude', 'longitude'])
        
        if len(geo_df) == 0:
            return None
        
        disaster_colors = {
            'fire': '#ef4444',
            'earthquake': '#f59e0b',
            'flood': '#3b82f6',
            'auto_accident': '#8b5cf6',
            'severe_storm': '#6b7280',
            'shooting': '#000000',
            'tornado': '#78350f',
            'hurricane': '#06b6d4',
            'other_disaster': '#ec4899'
        }
        
        points_data = []
        for _, row in geo_df.iterrows():
            points_data.append({
                'id': row['uri'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'disaster_type': row['disaster_type'],
                'severity': row['severity_level'],
                'location': row['location_mentioned'] or 'Not specified',
                'confidence': round(float(row['model_confidence']), 3),
                'help_request': bool(row['help_request']),
                'color': disaster_colors.get(row['disaster_type'], '#10b981'),
                'timestamp': row['processed_at'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        self.chart_data['geographic_data'] = {
            'center': {
                'lat': float(geo_df['latitude'].mean()),
                'lng': float(geo_df['longitude'].mean())
            },
            'points': points_data,
            'total_points': len(points_data)
        }
        
        return self.chart_data['geographic_data']

    def export_chart_data(self, filename: str = "chart_data.json"):
        """Export all chart data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.chart_data, f, indent=2, default=str)
        
        print(f"Chart data exported to {filename}")
        return self.chart_data

    def run_all_analyses(self):
        """Run all visualizations and analyses"""
        print("Starting comprehensive disaster data analysis...")
        
        # Generate all analyses
        self.generate_dashboard_metrics()
        self.disaster_type_distribution()
        self.severity_analysis()
        self.temporal_analysis()
        self.help_request_analysis()
        self.model_confidence_analysis()
        self.location_analysis()
        self.author_analysis()
        self.get_geographic_data()
        
        # Export all data
        chart_data = self.export_chart_data()
        
        print("\nAll analyses completed!")
        print("Chart data structure generated for shadcn/ui components")
        print("Available chart keys:", list(chart_data.keys()))
        
        return chart_data

if __name__ == "__main__":
    csv_file = "be_extracted_info_output.csv"
    
    try:
        visualizer = DisasterDataVisualizer(csv_file)
        chart_data = visualizer.run_all_analyses()
        
        # Print sample of the data structure
        print("\nSample metrics:", chart_data.get('metrics', {}))
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("Please update the csv_file path in the script.")
    except Exception as e:
        print(f"Error during analysis: {e}")