import pandas as pd
import re
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class EdgeFunctionLogAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the log analyzer with CSV file path
        """
        self.df = pd.read_csv("cw-logs.csv")
        self.analysis_results = {}
    
    def parse_log_messages(self):
        """
        Extract meaningful data from the log messages
        """
        print("Parsing log messages...")
        
        # Convert timestamp from nanoseconds to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'].astype(float) / 1e6, unit='ms')
        
        # Extract performance metrics
        self.df['success_count'] = self.df['event_message'].str.extract(r'Success: (\d+)')
        self.df['error_count'] = self.df['event_message'].str.extract(r'Errors: (\d+)')
        self.df['total_processing_time'] = self.df['event_message'].str.extract(r'Total: (\d+)ms')
        self.df['avg_classification_time'] = self.df['event_message'].str.extract(r'Avg classification: (\d+)ms')
        self.df['posts_per_minute'] = self.df['event_message'].str.extract(r'Posts/minute: (\d+\.?\d*)')
        
        # Extract classification results
        self.df['classification_result'] = self.df['event_message'].str.extract(r'Classification result: (\w+)')
        self.df['classification_time'] = self.df['event_message'].str.extract(r'Classification result: \w+ \((\d+)ms\)')
        
        # Extract individual classification times
        self.df['individual_classification_time'] = self.df['event_message'].str.extract(r'COMPLETED in: (\d+)ms')
        
        # Convert extracted columns to numeric
        numeric_columns = ['success_count', 'error_count', 'total_processing_time', 
                          'avg_classification_time', 'posts_per_minute', 
                          'classification_time', 'individual_classification_time']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Total log entries: {len(self.df)}")
        return self.df
    
    def analyze_performance(self):
        """
        Analyze performance metrics from the logs
        """
        print("\n=== PERFORMANCE ANALYSIS ===")
        
        # Get summary entries (FINISHED logs)
        summary_entries = self.df[self.df['event_message'].str.contains('FINISHED', na=False)]
        
        if len(summary_entries) > 0:
            total_success = summary_entries['success_count'].sum()
            total_errors = summary_entries['error_count'].sum()
            total_processed = total_success + total_errors
            
            print(f"Total tweets processed: {total_processed}")
            print(f"Successful classifications: {total_success}")
            print(f"Errors: {total_errors}")
            print(f"Success rate: {(total_success/total_processed*100):.2f}%" if total_processed > 0 else "N/A")
        
        # Analyze processing times
        if 'total_processing_time' in self.df.columns:
            valid_times = self.df['total_processing_time'].dropna()
            if len(valid_times) > 0:
                print(f"Average total processing time: {valid_times.mean():.2f}ms")
                print(f"Max processing time: {valid_times.max()}ms")
                print(f"Min processing time: {valid_times.min()}ms")
        
        # Analyze classification times
        classification_times = self.df['individual_classification_time'].dropna()
        if len(classification_times) > 0:
            print(f"Average individual classification time: {classification_times.mean():.2f}ms")
            print(f"Max classification time: {classification_times.max()}ms")
            print(f"Min classification time: {classification_times.min()}ms")
        
        # Analyze throughput
        posts_per_minute = self.df['posts_per_minute'].dropna()
        if len(posts_per_minute) > 0:
            print(f"Average posts per minute: {posts_per_minute.mean():.2f}")
            print(f"Max posts per minute: {posts_per_minute.max()}")
        
        return self.analysis_results
    
    def analyze_classification_distribution(self):
        """
        Analyze the distribution of classification results
        """
        print("\n=== CLASSIFICATION DISTRIBUTION ===")
        
        if 'classification_result' in self.df.columns:
            result_counts = self.df['classification_result'].value_counts()
            print("Classification Results:")
            for result, count in result_counts.items():
                if pd.notna(result):
                    print(f"  {result}: {count}")
        
        return self.analysis_results
    
    def analyze_timing_patterns(self):
        """
        Analyze timing patterns to identify rate limiting issues
        """
        print("\n=== TIMING PATTERNS ===")
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')
        
        # Calculate time differences between log entries
        self.df['time_diff'] = self.df['timestamp'].diff()
        
        # Convert time differences to seconds
        self.df['time_diff_seconds'] = self.df['time_diff'].dt.total_seconds()
        
        # Analyze gaps between classification operations
        classification_starts = self.df[self.df['event_message'].str.contains('START', na=False)]
        
        if len(classification_starts) > 1:
            start_times = classification_starts['timestamp']
            time_diffs = start_times.diff().dropna()
            
            if len(time_diffs) > 0:
                avg_gap = time_diffs.mean().total_seconds()
                min_gap = time_diffs.min().total_seconds()
                max_gap = time_diffs.max().total_seconds()
                
                print(f"Average gap between classifications: {avg_gap:.2f} seconds")
                print(f"Minimum gap: {min_gap:.2f} seconds")
                print(f"Maximum gap: {max_gap:.2f} seconds")
                
                # Check if gaps are too small (potential rate limiting)
                if avg_gap < 1.0:
                    print("WARNING: Very short gaps between requests - may trigger rate limiting")
        
        return self.analysis_results
    
    def estimate_token_usage(self):
        """
        Estimate token usage based on processing patterns
        """
        print("\n=== TOKEN USAGE ESTIMATION ===")
        
        # Get total processed tweets from summary entries
        summary_entries = self.df[self.df['event_message'].str.contains('FINISHED', na=False)]
        total_tweets = summary_entries['success_count'].sum() + summary_entries['error_count'].sum()
        
        if total_tweets > 0:
            # Estimate tokens per tweet (using your professor's estimates)
            token_estimates = {
                'optimistic': 15,
                'realistic': 30, 
                'conservative': 50
            }
            
            print(f"Total tweets processed in logs: {total_tweets}")
            print("\nEstimated token usage:")
            for scenario, tokens_per_tweet in token_estimates.items():
                total_tokens = total_tweets * tokens_per_tweet
                daily_capacity = 500000 / tokens_per_tweet
                print(f"  {scenario.capitalize()} ({tokens_per_tweet} tokens/tweet):")
                print(f"    - Total tokens used: {total_tokens:,}")
                print(f"    - Daily capacity: ~{daily_capacity:,.0f} tweets")
        
        return self.analysis_results
    
    def generate_recommendations(self):
        """
        Generate optimization recommendations
        """
        print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
        
        recommendations = []
        
        # Analyze current throughput
        posts_per_minute = self.df['posts_per_minute'].dropna()
        if len(posts_per_minute) > 0:
            current_throughput = posts_per_minute.mean()
            if current_throughput > 50:
                recommendations.append(
                    f"Current throughput ({current_throughput:.1f} posts/min) is good"
                )
            else:
                recommendations.append(
                    f"Current throughput ({current_throughput:.1f} posts/min) could be improved"
                )
        
        # Check for potential batching opportunities
        summary_entries = self.df[self.df['event_message'].str.contains('FINISHED', na=False)]
        if len(summary_entries) > 0:
            avg_batch_size = summary_entries['success_count'].mean()
            if avg_batch_size < 10:
                recommendations.append(
                    f"Consider batching more tweets (current avg: {avg_batch_size:.1f} per run)"
                )
        
        # Timing recommendations
        classification_times = self.df['individual_classification_time'].dropna()
        if len(classification_times) > 0:
            avg_class_time = classification_times.mean()
            recommendations.append(
                f"⏱Average classification time: {avg_class_time:.0f}ms"
            )
        
        # Rate limiting prevention
        recommendations.extend([
            "Add 1-2 second delay between API calls if not batching",
            "Implement tweet batching (process 10-50 tweets per API call)",
            "Use multiple API keys for different batches",
            "Monitor daily token usage to stay under 500k TPD limit"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return recommendations
    
    def create_visualizations(self):
        """
        Create visualizations for the analysis
        """
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Edge Function Performance Analysis')
        
        try:
            # Plot 1: Processing timeline
            if 'timestamp' in self.df.columns:
                timeline_df = self.df[self.df['event_message'].str.contains('START|FINISHED', na=False)].copy()
                if len(timeline_df) > 0:
                    axes[0,0].plot(timeline_df['timestamp'], range(len(timeline_df)), 'o-')
                    axes[0,0].set_title('Processing Timeline')
                    axes[0,0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Classification results distribution
            if 'classification_result' in self.df.columns:
                result_counts = self.df['classification_result'].value_counts()
                if len(result_counts) > 0:
                    result_counts.plot(kind='bar', ax=axes[0,1])
                    axes[0,1].set_title('Classification Results Distribution')
            
            # Plot 3: Processing time distribution
            if 'individual_classification_time' in self.df.columns:
                classification_times = self.df['individual_classification_time'].dropna()
                if len(classification_times) > 0:
                    axes[1,0].hist(classification_times, bins=20, alpha=0.7)
                    axes[1,0].set_title('Individual Classification Times')
                    axes[1,0].set_xlabel('Time (ms)')
            
            # Plot 4: Posts per minute over time
            if 'posts_per_minute' in self.df.columns:
                ppm_data = self.df[['timestamp', 'posts_per_minute']].dropna()
                if len(ppm_data) > 0:
                    axes[1,1].plot(ppm_data['timestamp'], ppm_data['posts_per_minute'], 'o-')
                    axes[1,1].set_title('Posts per Minute Over Time')
                    axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('edge_function_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualizations saved as 'edge_function_analysis.png'")
            
        except Exception as e:
            print(f"Could not generate all visualizations: {e}")
    
    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        print("\n" + "="*50)
        print("EDGE FUNCTION LOG ANALYSIS REPORT")
        print("="*50)
        
        self.parse_log_messages()
        self.analyze_performance()
        self.analyze_classification_distribution()
        self.analyze_timing_patterns()
        self.estimate_token_usage()
        self.generate_recommendations()
        
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Visualization error: {e}")
        
        print("\nAnalysis complete!")

def main():
    """
    Main function to run the analysis
    """
    csv_file_path = "classification_worker_logs.csv"  # Update this path
    
    try:
        analyzer = EdgeFunctionLogAnalyzer(csv_file_path)
        analyzer.generate_report()
        
        # Save processed data
        analyzer.df.to_csv('processed_edge_function_logs.csv', index=False)
        print("Processed data saved as 'processed_edge_function_logs.csv'")
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please update the csv_file_path variable with your actual file path")
    except Exception as e:
        print(f"Error analyzing logs: {e}")

if __name__ == "__main__":
    main()