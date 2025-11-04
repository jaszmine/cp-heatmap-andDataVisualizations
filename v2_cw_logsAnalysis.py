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
        self.df = pd.read_csv(csv_file_path)
        self.analysis_results = {}
    
    def parse_json_logs(self):
        """
        Parse the new JSON log format
        """
        print("Parsing JSON log messages...")
        
        # Convert timestamp from nanoseconds to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'].astype(float) / 1e6, unit='ms')
        
        # Parse JSON from event_message
        parsed_logs = []
        for _, row in self.df.iterrows():
            try:
                log_data = json.loads(row['event_message'])
                log_data['timestamp'] = row['timestamp']
                log_data['level'] = row.get('level', 'info')
                parsed_logs.append(log_data)
            except json.JSONDecodeError:
                # Skip non-JSON logs (like "booted" messages)
                continue
        
        # Create new DataFrame with parsed JSON data
        self.json_df = pd.DataFrame(parsed_logs)
        
        if len(self.json_df) > 0:
            print(f"Successfully parsed {len(self.json_df)} JSON log entries")
            print(f"Log types found: {self.json_df['type'].value_counts().to_dict()}")
        else:
            print("No JSON logs found to parse")
            self.json_df = pd.DataFrame()
        
        return self.json_df
    
    def analyze_performance(self):
        """
        Analyze performance metrics from JSON logs
        """
        print("\n=== PERFORMANCE ANALYSIS ===")
        
        if self.json_df.empty:
            print("No JSON data to analyze")
            return self.analysis_results
        
        # Get worker summary entries
        summary_entries = self.json_df[self.json_df['type'] == 'worker_summary']
        
        if len(summary_entries) > 0:
            total_success = summary_entries['success_count'].sum()
            total_errors = summary_entries['error_count'].sum()
            total_processed = total_success + total_errors
            
            print(f"Total runs analyzed: {len(summary_entries)}")
            print(f"Total tweets processed: {total_processed}")
            print(f"Successful classifications: {total_success}")
            print(f"Errors: {total_errors}")
            print(f"Success rate: {(total_success/total_processed*100):.2f}%" if total_processed > 0 else "N/A")
            
            # Performance metrics from summaries
            avg_total_time = summary_entries['total_time_ms'].mean()
            avg_classification_time = summary_entries['avg_classification_time_ms'].mean()
            avg_posts_per_minute = summary_entries['posts_per_minute'].astype(float).mean()
            
            print(f"Average total processing time: {avg_total_time:.2f}ms")
            print(f"Average classification time: {avg_classification_time:.2f}ms")
            print(f"Average posts per minute: {avg_posts_per_minute:.2f}")
        
        return self.analysis_results
    
    def analyze_classification_distribution(self):
        """
        Analyze the distribution of classification results from JSON logs
        """
        print("\n=== CLASSIFICATION DISTRIBUTION ===")
        
        if self.json_df.empty:
            print("No JSON data to analyze")
            return self.analysis_results
        
        # Get classification result entries
        classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
        
        if len(classification_entries) > 0:
            result_counts = classification_entries['disaster_type'].value_counts()
            print("Classification Results Distribution:")
            for result, count in result_counts.items():
                percentage = (count / len(classification_entries)) * 100
                print(f"  {result}: {count} ({percentage:.1f}%)")
            
            # Action distribution
            action_counts = classification_entries['action'].value_counts()
            print("\nAction Distribution:")
            for action, count in action_counts.items():
                if pd.notna(action):
                    print(f"  {action}: {count}")
        
        return self.analysis_results
    
    def analyze_timing_patterns(self):
        """
        Analyze timing patterns to identify rate limiting issues
        """
        print("\n=== TIMING PATTERNS ===")
        
        if self.json_df.empty:
            print("No JSON data to analyze")
            return self.analysis_results
        
        # Sort by timestamp
        self.json_df = self.json_df.sort_values('timestamp')
        
        # Analyze classification times
        classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
        if len(classification_entries) > 0:
            classification_times = classification_entries['classification_time'].dropna()
            print(f"Classification Time Stats:")
            print(f"  Average: {classification_times.mean():.2f}ms")
            print(f"  Min: {classification_times.min()}ms")
            print(f"  Max: {classification_times.max()}ms")
            print(f"  Std Dev: {classification_times.std():.2f}ms")
        
        # Analyze gaps between classification operations
        if len(classification_entries) > 1:
            classification_entries = classification_entries.sort_values('timestamp')
            time_diffs = classification_entries['timestamp'].diff().dropna()
            
            if len(time_diffs) > 0:
                avg_gap = time_diffs.mean().total_seconds()
                min_gap = time_diffs.min().total_seconds()
                max_gap = time_diffs.max().total_seconds()
                
                print(f"\nRequest Timing Analysis:")
                print(f"  Average gap between requests: {avg_gap:.2f} seconds")
                print(f"  Minimum gap: {min_gap:.2f} seconds")
                print(f"  Maximum gap: {max_gap:.2f} seconds")
                
                # Check if gaps are appropriate
                if avg_gap < 1.0:
                    print(" WARNING: Gaps may be too short - risk of rate limiting")
                elif avg_gap >= 1.4 and avg_gap <= 1.6:
                    print(" Gaps are optimal (1.5s delays working correctly)")
                else:
                    print(f" Current gap: {avg_gap:.2f}s")
        
        return self.analysis_results
    
    def analyze_model_confidence(self):
        """
        Analyze model confidence scores
        """
        print("\n=== MODEL CONFIDENCE ANALYSIS ===")
        
        if self.json_df.empty:
            print("No JSON data to analyze")
            return self.analysis_results
        
        classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
        
        if len(classification_entries) > 0:
            confidence_scores = classification_entries['model_confidence'].dropna()
            print(f"Model Confidence Stats:")
            print(f"  Average: {confidence_scores.mean():.3f}")
            print(f"  Min: {confidence_scores.min():.3f}")
            print(f"  Max: {confidence_scores.max():.3f}")
            
            # Confidence by disaster type
            confidence_by_type = classification_entries.groupby('disaster_type')['model_confidence'].mean()
            print(f"\nAverage Confidence by Disaster Type:")
            for disaster_type, confidence in confidence_by_type.items():
                print(f"  {disaster_type}: {confidence:.3f}")
        
        return self.analysis_results
    
    def estimate_token_usage(self):
        """
        Estimate token usage based on processing patterns
        """
        print("\n=== TOKEN USAGE ESTIMATION ===")
        
        if self.json_df.empty:
            print("No JSON data to analyze")
            return self.analysis_results
        
        # Get total processed tweets from classification results
        classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
        total_tweets = len(classification_entries)
        
        if total_tweets > 0:
            # Estimate tokens per tweet
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
                print(f"    - Current dataset coverage: {(35000/daily_capacity*100):.1f}% of 35k tweets")
        
        return self.analysis_results
    
    def generate_recommendations(self):
        """
        Generate optimization recommendations based on JSON logs
        """
        print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
        
        recommendations = []
        
        if self.json_df.empty:
            recommendations.append("No JSON log data available for analysis")
            return recommendations
        
        # Analyze current throughput
        summary_entries = self.json_df[self.json_df['type'] == 'worker_summary']
        if len(summary_entries) > 0:
            current_throughput = summary_entries['posts_per_minute'].astype(float).mean()
            if current_throughput > 50:
                recommendations.append(f"Current throughput ({current_throughput:.1f} posts/min) is good")
            else:
                recommendations.append(f"Throughput ({current_throughput:.1f} posts/min) could be improved")
        
        # Analyze classification distribution
        classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
        if len(classification_entries) > 0:
            not_relevant_pct = (classification_entries['disaster_type'] == 'not_relevant').mean() * 100
            recommendations.append(f"{not_relevant_pct:.1f}% of posts are 'not_relevant'")
            
            if not_relevant_pct > 80:
                recommendations.append("Consider filtering more aggressively before classification")
        
        # Check for errors
        error_entries = self.json_df[self.json_df['type'] == 'classification_error']
        if len(error_entries) > 0:
            recommendations.append(f"Found {len(error_entries)} classification errors - review error logs")
        
        # Rate limiting assessment
        classification_entries_sorted = classification_entries.sort_values('timestamp')
        if len(classification_entries_sorted) > 1:
            avg_gap = classification_entries_sorted['timestamp'].diff().mean().total_seconds()
            if 1.4 <= avg_gap <= 1.6:
                recommendations.append("1.5s delays are working correctly")
            else:
                recommendations.append(f"Current delay: {avg_gap:.2f}s (target: 1.5s)")
        
        # General recommendations
        recommendations.extend([
            "Consider increasing batch size from 6 to 10-15",
            "Monitor token usage as you scale frequency",
            "Current 5-minute cron interval is appropriate",
            "Review disaster_type patterns for model tuning"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        return recommendations
    
    def create_visualizations(self):
        """
        Create visualizations for the JSON log analysis
        """
        print("\nGenerating visualizations...")
        
        if self.json_df.empty:
            print("No JSON data for visualizations")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Edge Function Performance Analysis (JSON Logs)')
        
        try:
            # Plot 1: Classification results distribution
            classification_entries = self.json_df[self.json_df['type'] == 'classification_result']
            if len(classification_entries) > 0:
                result_counts = classification_entries['disaster_type'].value_counts()
                result_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
                axes[0,0].set_title('Classification Results Distribution')
                axes[0,0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Classification time distribution
            if len(classification_entries) > 0:
                classification_times = classification_entries['classification_time'].dropna()
                axes[0,1].hist(classification_times, bins=20, alpha=0.7, color='lightgreen')
                axes[0,1].set_title('Classification Time Distribution')
                axes[0,1].set_xlabel('Time (ms)')
                axes[0,1].set_ylabel('Frequency')
            
            # Plot 3: Model confidence distribution
            if len(classification_entries) > 0:
                confidence_scores = classification_entries['model_confidence'].dropna()
                axes[1,0].hist(confidence_scores, bins=20, alpha=0.7, color='orange')
                axes[1,0].set_title('Model Confidence Distribution')
                axes[1,0].set_xlabel('Confidence Score')
                axes[1,0].set_ylabel('Frequency')
            
            # Plot 4: Processing timeline
            worker_starts = self.json_df[self.json_df['type'] == 'worker_start']
            if len(worker_starts) > 0:
                axes[1,1].plot(worker_starts['timestamp'], range(len(worker_starts)), 'o-', color='red')
                axes[1,1].set_title('Worker Execution Timeline')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].set_ylabel('Run Number')
            
            plt.tight_layout()
            plt.savefig('edge_function_json_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualizations saved as 'edge_function_json_analysis.png'")
            
        except Exception as e:
            print(f"Could not generate all visualizations: {e}")
    
    def generate_report(self):
        """
        Generate comprehensive analysis report for JSON logs
        """
        print("\n" + "="*50)
        print("EDGE FUNCTION JSON LOG ANALYSIS REPORT")
        print("="*50)
        
        self.parse_json_logs()
        self.analyze_performance()
        self.analyze_classification_distribution()
        self.analyze_timing_patterns()
        self.analyze_model_confidence()
        self.estimate_token_usage()
        self.generate_recommendations()
        
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Visualization error: {e}")
        
        print("\nJSON log analysis complete!")

def main():
    """
    Main function to run the analysis
    """
    csv_file_path = "cw-logs.csv"  # Your CSV file name
    
    try:
        analyzer = EdgeFunctionLogAnalyzer(csv_file_path)
        analyzer.generate_report()
        
        # Save processed JSON data
        if not analyzer.json_df.empty:
            analyzer.json_df.to_csv('processed_json_logs.csv', index=False)
            print("Processed JSON data saved as 'processed_json_logs.csv'")
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        print("Please update the csv_file_path variable with your actual file path")
    except Exception as e:
        print(f"Error analyzing logs: {e}")

if __name__ == "__main__":
    main()