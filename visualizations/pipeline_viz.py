import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Add this BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PipelineVisualizer:
    def __init__(self, data_path):
        """
        Initialize the visualizer with pipeline diagnostics data
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = 'static/images'
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the pipeline diagnostics data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {len(self.df)} records")
            
            # Basic data cleaning
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['total_pipeline_time_ms'] = pd.to_numeric(self.df['total_pipeline_time_ms'], errors='coerce')
            self.df['problem_length_words'] = pd.to_numeric(self.df['problem_length_words'], errors='coerce')
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        eda_summary = {
            'total_records': len(self.df),
            'date_range': {
                'start': self.df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': self.df['timestamp'].max().strftime('%Y-%m-%d')
            },
            'success_rate': round((self.df['solver_success'].mean() * 100), 2),
            'error_rate': round((self.df['error_occurred'].mean() * 100), 2),
            'avg_pipeline_time': round(self.df['total_pipeline_time_ms'].mean(), 2),
            'topics_distribution': self.df['topic'].value_counts().to_dict(),
            'difficulty_spread': self.df['difficulty_level'].value_counts().to_dict(),
            'model_versions': self.df['model_version'].value_counts().to_dict(),
            'common_errors': self.df['error_type'].value_counts().head(5).to_dict()
        }
        
        # Add statistical summaries
        eda_summary['pipeline_time_stats'] = {
            'mean': round(self.df['total_pipeline_time_ms'].mean(), 2),
            'median': round(self.df['total_pipeline_time_ms'].median(), 2),
            'std': round(self.df['total_pipeline_time_ms'].std(), 2),
            'min': round(self.df['total_pipeline_time_ms'].min(), 2),
            'max': round(self.df['total_pipeline_time_ms'].max(), 2)
        }
        
        return eda_summary
    
    def create_pipeline_funnel(self):
        """Create pipeline success funnel visualization"""
        plt.figure(figsize=(10, 6))
        
        # Calculate success rates at each stage
        stages = ['Text Cleaning', 'Equation Extraction', 'Embedding', 'Solver']
        success_rates = [
            self.df['text_cleaning_success'].mean() * 100,
            self.df['equation_extraction_success'].mean() * 100,
            self.df['embedding_success'].mean() * 100,
            self.df['solver_success'].mean() * 100
        ]
        
        # Create funnel plot
        plt.barh(stages, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.xlabel('Success Rate (%)')
        plt.title('Pipeline Success Funnel')
        
        # Add value labels
        for i, v in enumerate(success_rates):
            plt.text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/pipeline_funnel.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_topic_time_boxplot(self):
        """Create boxplot of topic vs total pipeline time"""
        plt.figure(figsize=(12, 6))
        
        # Filter out extreme outliers for better visualization
        Q1 = self.df['total_pipeline_time_ms'].quantile(0.25)
        Q3 = self.df['total_pipeline_time_ms'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = self.df[(self.df['total_pipeline_time_ms'] <= Q3 + 1.5 * IQR)]
        
        sns.boxplot(data=filtered_df, x='topic', y='total_pipeline_time_ms', palette='viridis')
        plt.title('Topic vs Total Pipeline Time')
        plt.xlabel('Topic')
        plt.ylabel('Pipeline Time (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/topic_time_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_difficulty_success_barchart(self):
        """Create bar chart of difficulty level vs success rate"""
        plt.figure(figsize=(10, 6))
        
        # Calculate success rate by difficulty
        success_by_difficulty = self.df.groupby('difficulty_level')['solver_success'].mean() * 100
        
        success_by_difficulty.plot(kind='bar', color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
        plt.title('Difficulty Level vs Solver Success Rate')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(success_by_difficulty):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/difficulty_success_barchart.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_model_comparison(self):
        """Create model version comparison chart"""
        plt.figure(figsize=(10, 6))
        
        # Group by model version and calculate metrics
        model_stats = self.df.groupby('model_version').agg({
            'solver_success': 'mean',
            'total_pipeline_time_ms': 'mean',
            'problem_id': 'count'
        }).round(3)
        
        # Create comparison plot
        x = np.arange(len(model_stats))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Success rate bars
        bars1 = ax1.bar(x - width/2, model_stats['solver_success'] * 100, 
                       width, label='Success Rate (%)', color='lightblue')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        
        # Pipeline time line
        ax2 = ax1.twinx()
        line = ax2.plot(x, model_stats['total_pipeline_time_ms'], 
                       color='red', marker='o', linewidth=2, 
                       label='Avg Pipeline Time (ms)')
        ax2.set_ylabel('Pipeline Time (ms)')
        
        ax1.set_xlabel('Model Version')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_stats.index)
        ax1.set_title('Model Version Comparison')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/model_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_error_frequency(self):
        """Create error type frequency chart"""
        plt.figure(figsize=(10, 6))
        
        # Get error frequency (excluding NaN)
        error_counts = self.df['error_type'].value_counts().head(10)
        
        # Create horizontal bar chart
        error_counts.plot(kind='barh', color='lightcoral')
        plt.title('Top 10 Error Types by Frequency')
        plt.xlabel('Count')
        
        # Add value labels
        for i, v in enumerate(error_counts):
            plt.text(v + 0.1, i, str(v), va='center')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/error_frequency.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for timing and complexity metrics"""
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['problem_length_words', 'equation_count', 
                       'extraction_time_ms', 'embedding_time_ms', 
                       'solver_time_ms', 'total_pipeline_time_ms']
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Correlation Heatmap: Timing & Complexity Metrics')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/correlation_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def perform_regression_analysis(self):
        """Perform regression: Problem Length → Total Pipeline Time"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Prepare data
        X = self.df[['problem_length_words']].dropna()
        y = self.df.loc[X.index, 'total_pipeline_time_ms']
        
        # Remove outliers for better regression
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Perform regression
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        y_pred = model.predict(X_clean)
        
        # Create regression plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X_clean, y_clean, alpha=0.6, color='blue', label='Data points')
        plt.plot(X_clean, y_pred, color='red', linewidth=2, label='Regression line')
        
        plt.xlabel('Problem Length (words)')
        plt.ylabel('Total Pipeline Time (ms)')
        plt.title(f'Regression: Problem Length → Pipeline Time\nR² = {r2_score(y_clean, y_pred):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/regression_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Regression results
        regression_results = {
            'r_squared': round(r2_score(y_clean, y_pred), 4),
            'coefficient': round(model.coef_[0], 4),
            'intercept': round(model.intercept_, 2),
            'equation': f"Pipeline Time = {model.intercept_:.2f} + {model.coef_[0]:.4f} * Problem Length",
            'interpretation': "For each additional word, pipeline time increases by approximately {:.2f} ms".format(model.coef_[0])
        }
        
        return regression_results, filename.replace('static/', '')
    
    def perform_hypothesis_test(self):
        """Perform ANOVA: Does Difficulty Level affect Total Pipeline Time?"""
        # Group data by difficulty level
        groups = []
        for level in sorted(self.df['difficulty_level'].unique()):
            group_data = self.df[self.df['difficulty_level'] == level]['total_pipeline_time_ms'].dropna()
            groups.append(group_data)
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        difficulty_data = [self.df[self.df['difficulty_level'] == level]['total_pipeline_time_ms'] 
                          for level in sorted(self.df['difficulty_level'].unique())]
        
        plt.boxplot(difficulty_data, labels=sorted(self.df['difficulty_level'].unique()))
        plt.xlabel('Difficulty Level')
        plt.ylabel('Total Pipeline Time (ms)')
        plt.title(f'ANOVA: Difficulty Level vs Pipeline Time\np-value = {p_value:.4f}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/hypothesis_test.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Hypothesis test results
        hypothesis_results = {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_value, 6),
            'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
            'null_hypothesis': 'Mean pipeline time is the same across all difficulty levels',
            'alternative_hypothesis': 'At least one difficulty level has different mean pipeline time',
            'conclusion': 'Reject H₀ - Significant difference exists' if p_value < 0.05 else 'Fail to reject H₀ - No significant difference'
        }
        
        return hypothesis_results, filename.replace('static/', '')
    
    def generate_insights(self, eda_summary, regression_results, hypothesis_results):
        """Generate automated insights from the analysis"""
        insights = []
        
        # Success rate insight
        success_rate = eda_summary['success_rate']
        if success_rate > 80:
            insights.append("🎯 <strong>High Success Rate:</strong> Pipeline is performing well with {:.1f}% overall success".format(success_rate))
        elif success_rate > 60:
            insights.append("⚠️ <strong>Moderate Success Rate:</strong> Pipeline has room for improvement at {:.1f}% success".format(success_rate))
        else:
            insights.append("🚨 <strong>Low Success Rate:</strong> Pipeline needs attention with only {:.1f}% success".format(success_rate))
        
        # Regression insight
        if regression_results['r_squared'] > 0.3:
            insights.append("📈 <strong>Strong Relationship:</strong> Problem length significantly affects pipeline time (R² = {:.3f})".format(regression_results['r_squared']))
        else:
            insights.append("📊 <strong>Weak Relationship:</strong> Problem length has limited impact on pipeline time (R² = {:.3f})".format(regression_results['r_squared']))
        
        # Hypothesis test insight
        if hypothesis_results['p_value'] < 0.05:
            insights.append("🔬 <strong>Significant Difference:</strong> Difficulty levels significantly affect pipeline time (p = {:.4f})".format(hypothesis_results['p_value']))
        else:
            insights.append("📊 <strong>No Significant Difference:</strong> Difficulty levels don't significantly affect pipeline time (p = {:.4f})".format(hypothesis_results['p_value']))
        
        # Error analysis insight
        most_common_error = list(eda_summary['common_errors'].keys())[0] if eda_summary['common_errors'] else 'None'
        insights.append("🛠️ <strong>Common Error:</strong> Most frequent error type is '{}'".format(most_common_error))
        
        # Performance insight
        avg_time = eda_summary['avg_pipeline_time']
        if avg_time < 1000:
            insights.append("⚡ <strong>Good Performance:</strong> Average pipeline time is efficient at {:.0f} ms".format(avg_time))
        else:
            insights.append("🐌 <strong>Performance Concern:</strong> Average pipeline time is high at {:.0f} ms".format(avg_time))
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Pipeline Diagnostics Analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Generate EDA summary
        print("📊 Generating EDA summary...")
        eda_summary = self.generate_eda_summary()
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        visuals = {
            'funnel_plot': self.create_pipeline_funnel(),
            'topic_time_plot': self.create_topic_time_boxplot(),
            'difficulty_success_plot': self.create_difficulty_success_barchart(),
            'model_comparison_plot': self.create_model_comparison(),
            'error_frequency_plot': self.create_error_frequency(),
            'correlation_heatmap': self.create_correlation_heatmap()
        }
        
        # Perform statistical analyses
        print("📈 Performing regression analysis...")
        regression_results, regression_plot = self.perform_regression_analysis()
        visuals['regression_plot'] = regression_plot
        
        print("🔬 Performing hypothesis testing...")
        hypothesis_results, hypothesis_plot = self.perform_hypothesis_test()
        visuals['hypothesis_plot'] = hypothesis_plot
        
        # Generate insights
        print("💡 Generating insights...")
        insights = self.generate_insights(eda_summary, regression_results, hypothesis_results)
        
        # Compile final results
        self.results = {
            'overview': {
                'total_records': eda_summary['total_records'],
                'success_rate': eda_summary['success_rate'],
                'avg_pipeline_time': eda_summary['avg_pipeline_time'],
                'error_rate': eda_summary['error_rate']
            },
            'eda': {
                'topics_distribution': str(eda_summary['topics_distribution']),
                'difficulty_spread': str(eda_summary['difficulty_spread']),
                'model_versions': str(eda_summary['model_versions']),
                'common_errors': str(eda_summary['common_errors'])
            },
            'visuals': visuals,
            'regression': regression_results,
            'hypothesis': hypothesis_results,
            'insights': '<br>'.join(insights)
        }
        
        print("✅ Analysis completed successfully!")
        return self.results

# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = PipelineVisualizer('static/datasets/pipeline_diagnostics.csv')
    
    # Run complete analysis
    results = visualizer.run_complete_analysis()
    
    if results:
        print(f"📁 Generated {len(results['visuals'])} visualizations")
        print(f"📊 EDA Summary: {results['overview']}")
    else:
        print("❌ Analysis failed!")