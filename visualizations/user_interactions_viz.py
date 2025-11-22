import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UserInteractionsVisualizer:
    def __init__(self, data_path):
        """
        Initialize the visualizer with user interactions data
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = 'static/images'
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the user interactions data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {len(self.df)} records")
            
            # Basic data cleaning
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['response_time_seconds'] = pd.to_numeric(self.df['response_time_seconds'], errors='coerce')
            self.df['user_accuracy'] = pd.to_numeric(self.df['user_accuracy'], errors='coerce')
            self.df['confidence_rating'] = pd.to_numeric(self.df['confidence_rating'], errors='coerce')
            self.df['study_hours_per_week'] = pd.to_numeric(self.df['study_hours_per_week'], errors='coerce')
            self.df['previous_math_score'] = pd.to_numeric(self.df['previous_math_score'], errors='coerce')
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        eda_summary = {
            'total_interactions': len(self.df),
            'unique_users': self.df['user_id'].nunique(),
            'unique_problems': self.df['problem_id'].nunique(),
            'date_range': {
                'start': self.df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': self.df['timestamp'].max().strftime('%Y-%m-%d')
            },
            'avg_accuracy': round(self.df['user_accuracy'].mean() * 100, 2),
            'avg_response_time': round(self.df['response_time_seconds'].mean(), 2),
            'hint_usage_rate': round((self.df['hint_used'] == 'yes').mean() * 100, 2),
            'topics_distribution': self.df['topic'].value_counts().to_dict(),
            'grade_level_distribution': self.df['grade_level'].value_counts().to_dict(),
            'time_of_day_distribution': self.df['time_of_day'].value_counts().to_dict(),
            'avg_confidence': round(self.df['confidence_rating'].mean(), 2)
        }
        
        # Add statistical summaries
        eda_summary['response_time_stats'] = {
            'mean': round(self.df['response_time_seconds'].mean(), 2),
            'median': round(self.df['response_time_seconds'].median(), 2),
            'std': round(self.df['response_time_seconds'].std(), 2),
            'min': round(self.df['response_time_seconds'].min(), 2),
            'max': round(self.df['response_time_seconds'].max(), 2)
        }
        
        return eda_summary
    
    def create_study_hours_accuracy_scatter(self):
        """Visualization 1: Study Hours vs Accuracy with Trend Line"""
        plt.figure(figsize=(12, 6))
        
        # Filter out missing values
        plot_df = self.df[['study_hours_per_week', 'user_accuracy']].dropna()
        
        # Create scatter plot
        plt.scatter(plot_df['study_hours_per_week'], 
                   plot_df['user_accuracy'], 
                   alpha=0.6, 
                   c=plot_df['user_accuracy'], 
                   cmap='viridis',
                   s=50,
                   edgecolors='black',
                   linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(plot_df['study_hours_per_week'], plot_df['user_accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(plot_df['study_hours_per_week'].sort_values(), 
                p(plot_df['study_hours_per_week'].sort_values()), 
                "r--", 
                linewidth=2, 
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        
        plt.xlabel('Study Hours per Week')
        plt.ylabel('User Accuracy')
        plt.title('Study Hours vs User Accuracy with Trend Analysis')
        plt.colorbar(label='Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/study_hours_accuracy.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_time_of_day_performance_heatmap(self):
        """Visualization 2: Time of Day vs Day of Week Performance Heatmap"""
        plt.figure(figsize=(12, 6))
        
        # Create pivot table for heatmap
        heatmap_data = self.df.groupby(['time_of_day', 'day_of_week'])['user_accuracy'].mean().unstack(fill_value=0)
        
        # Define custom order for time_of_day
        time_order = ['morning', 'afternoon', 'evening', 'night']
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Reindex to maintain order
        heatmap_data = heatmap_data.reindex(time_order, fill_value=0)
        heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns], fill_value=0)
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlGn', 
                   center=0.5,
                   cbar_kws={'label': 'Average Accuracy'},
                   linewidths=0.5,
                   linecolor='gray')
        
        plt.title('User Performance Heatmap: Time of Day vs Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Time of Day')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/time_day_performance_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_difficulty_response_time_violin(self):
        """Visualization 3: Difficulty Level vs Response Time Violin Plot"""
        plt.figure(figsize=(12, 6))
        
        # Filter outliers for better visualization
        Q1 = self.df['response_time_seconds'].quantile(0.25)
        Q3 = self.df['response_time_seconds'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = self.df[(self.df['response_time_seconds'] >= Q1 - 1.5 * IQR) & 
                              (self.df['response_time_seconds'] <= Q3 + 1.5 * IQR)]
        
        # Create violin plot
        sns.violinplot(data=filtered_df, 
                      x='difficulty_level', 
                      y='response_time_seconds',
                      palette='muted',
                      inner='quartile')
        
        plt.title('Response Time Distribution Across Difficulty Levels')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Response Time (seconds)')
        
        # Add mean values as text
        means = filtered_df.groupby('difficulty_level')['response_time_seconds'].mean()
        for i, (level, mean_val) in enumerate(means.items()):
            plt.text(i, mean_val, f'μ={mean_val:.0f}s', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/difficulty_response_violin.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_hint_usage_impact_comparison(self):
        """Visualization 4: Hint Usage Impact on Accuracy and Response Time"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy comparison
        hint_accuracy = self.df.groupby('hint_used')['user_accuracy'].mean()
        colors = ['#FF6B6B', '#4ECDC4']
        axes[0].bar(hint_accuracy.index, hint_accuracy.values, color=colors)
        axes[0].set_title('Average Accuracy by Hint Usage')
        axes[0].set_xlabel('Hint Used')
        axes[0].set_ylabel('Average Accuracy')
        axes[0].set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(hint_accuracy.values):
            axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        # Response time comparison
        hint_time = self.df.groupby('hint_used')['response_time_seconds'].mean()
        axes[1].bar(hint_time.index, hint_time.values, color=colors)
        axes[1].set_title('Average Response Time by Hint Usage')
        axes[1].set_xlabel('Hint Used')
        axes[1].set_ylabel('Average Response Time (seconds)')
        
        # Add value labels
        for i, v in enumerate(hint_time.values):
            axes[1].text(i, v + 5, f'{v:.0f}s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/hint_usage_impact.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_grade_level_topic_performance(self):
        """Visualization 5: Grade Level Performance Across Topics"""
        plt.figure(figsize=(14, 7))
        
        # Calculate average accuracy by grade level and topic
        performance = self.df.groupby(['grade_level', 'topic'])['user_accuracy'].mean().unstack(fill_value=0)
        
        # Create grouped bar chart
        performance.plot(kind='bar', width=0.8, colormap='tab10')
        
        plt.title('Student Performance by Grade Level and Topic')
        plt.xlabel('Grade Level')
        plt.ylabel('Average Accuracy')
        plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/grade_topic_performance.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_confidence_accuracy_relationship(self):
        """Visualization 6: Confidence Rating vs Actual Accuracy"""
        plt.figure(figsize=(12, 6))
        
        # Group by confidence rating and calculate metrics
        confidence_groups = self.df.groupby('confidence_rating').agg({
            'user_accuracy': ['mean', 'count']
        }).reset_index()
        confidence_groups.columns = ['confidence_rating', 'avg_accuracy', 'count']
        
        # Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Accuracy line
        color = '#4ECDC4'
        ax1.set_xlabel('Confidence Rating')
        ax1.set_ylabel('Average Accuracy', color=color)
        line1 = ax1.plot(confidence_groups['confidence_rating'], 
                        confidence_groups['avg_accuracy'], 
                        color=color, 
                        marker='o', 
                        linewidth=3,
                        markersize=10,
                        label='Average Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Count bars
        ax2 = ax1.twinx()
        color = '#FF6B6B'
        ax2.set_ylabel('Number of Responses', color=color)
        bars = ax2.bar(confidence_groups['confidence_rating'], 
                      confidence_groups['count'], 
                      alpha=0.3, 
                      color=color,
                      label='Response Count')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Confidence Rating vs Actual Accuracy (Calibration Analysis)')
        
        # Combine legends
        lines = line1
        labels = [l.get_label() for l in lines]
        ax1.legend(lines + [bars], labels + ['Response Count'], loc='upper left')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/confidence_accuracy_relationship.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename.replace('static/', '')
    
    def perform_regression_analysis(self):
        """Perform Multiple Linear Regression: Study Hours + Previous Score → Accuracy"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Prepare data
        features = ['study_hours_per_week', 'previous_math_score']
        X = self.df[features].dropna()
        y = self.df.loc[X.index, 'user_accuracy']
        
        # Perform regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Actual vs Predicted
        axes[0].scatter(y, y_pred, alpha=0.6, color='blue', s=50, edgecolors='black', linewidth=0.5)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Accuracy')
        axes[0].set_ylabel('Predicted Accuracy')
        axes[0].set_title(f'Actual vs Predicted Accuracy\nR² = {r2_score(y, y_pred):.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Feature Coefficients
        coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=True)
        
        colors_coef = ['red' if x < 0 else 'green' for x in coefficients['Coefficient']]
        axes[1].barh(coefficients['Feature'], coefficients['Coefficient'], color=colors_coef)
        axes[1].set_xlabel('Coefficient Value')
        axes[1].set_title('Feature Importance in Predicting Accuracy')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/interaction_regression_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Regression results
        regression_results = {
            'r_squared': round(r2_score(y, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y, y_pred)), 4),
            'coefficients': {
                'study_hours_per_week': round(model.coef_[0], 4),
                'previous_math_score': round(model.coef_[1], 4)
            },
            'intercept': round(model.intercept_, 4),
            'equation': f"Accuracy = {model.intercept_:.4f} + {model.coef_[0]:.4f}*StudyHours + {model.coef_[1]:.4f}*PreviousScore",
            'interpretation': f"Study hours contributes {model.coef_[0]:.4f} and previous score contributes {model.coef_[1]:.4f} to accuracy prediction"
        }
        
        return regression_results, filename.replace('static/', '')
    
    def perform_hypothesis_test(self):
        """Perform T-Test: Does Hint Usage Significantly Affect User Accuracy?"""
        # Separate data into two groups
        hint_yes = self.df[self.df['hint_used'] == 'yes']['user_accuracy'].dropna()
        hint_no = self.df[self.df['hint_used'] == 'no']['user_accuracy'].dropna()
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(hint_yes, hint_no)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Box plot comparison
        data_to_plot = [hint_no, hint_yes]
        box = plt.boxplot(data_to_plot, 
                         labels=['No Hint', 'Hint Used'],
                         patch_artist=True,
                         notch=True,
                         showmeans=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add mean values
        means = [hint_no.mean(), hint_yes.mean()]
        for i, mean in enumerate(means, 1):
            plt.text(i, mean, f'μ={mean:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.ylabel('User Accuracy')
        plt.title(f'T-Test: Impact of Hint Usage on User Accuracy\nt-statistic = {t_stat:.4f}, p-value = {p_value:.6f}')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add significance annotation
        if p_value < 0.05:
            plt.text(1.5, max(hint_no.max(), hint_yes.max()) * 0.95, 
                    '* Statistically Significant (p < 0.05)', 
                    ha='center', fontsize=11, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        filename = f"{self.output_dir}/interactions_hypothesis_test.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Hypothesis test results
        hypothesis_results = {
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
            'null_hypothesis': 'Hint usage has no effect on user accuracy (means are equal)',
            'alternative_hypothesis': 'Hint usage significantly affects user accuracy (means are different)',
            'conclusion': 'Reject H₀ - Hint usage significantly impacts accuracy' if p_value < 0.05 else 'Fail to reject H₀ - No significant impact of hint usage',
            'mean_no_hint': round(hint_no.mean(), 4),
            'mean_with_hint': round(hint_yes.mean(), 4),
            'difference': round(abs(hint_yes.mean() - hint_no.mean()), 4)
        }
        
        return hypothesis_results, filename.replace('static/', '')
    
    def generate_insights(self, eda_summary, regression_results, hypothesis_results):
        """Generate automated insights from the analysis"""
        insights = []
        
        # Accuracy insight
        avg_accuracy = eda_summary['avg_accuracy']
        if avg_accuracy > 80:
            insights.append(f"🎯 <strong>High User Accuracy:</strong> Students achieving {avg_accuracy:.1f}% average accuracy - excellent performance")
        elif avg_accuracy > 60:
            insights.append(f"⚠️ <strong>Moderate User Accuracy:</strong> Students at {avg_accuracy:.1f}% accuracy - room for improvement")
        else:
            insights.append(f"🚨 <strong>Low User Accuracy:</strong> Students struggling at {avg_accuracy:.1f}% accuracy - intervention needed")
        
        # Regression insight
        if regression_results['r_squared'] > 0.3:
            insights.append(f"📈 <strong>Strong Predictors:</strong> Study hours and previous scores strongly predict accuracy (R² = {regression_results['r_squared']:.3f})")
        else:
            insights.append(f"📊 <strong>Weak Predictors:</strong> Additional factors beyond study time affect performance (R² = {regression_results['r_squared']:.3f})")
        
        # Hypothesis test insight
        if hypothesis_results['p_value'] < 0.05:
            diff_direction = "improves" if hypothesis_results['mean_with_hint'] > hypothesis_results['mean_no_hint'] else "reduces"
            insights.append(f"🔬 <strong>Hint Impact:</strong> Using hints {diff_direction} accuracy by {hypothesis_results['difference']:.2%} (p = {hypothesis_results['p_value']:.4f})")
        else:
            insights.append(f"📊 <strong>No Hint Effect:</strong> Hints don't significantly change accuracy (p = {hypothesis_results['p_value']:.4f})")
        
        # Hint usage insight
        hint_rate = eda_summary['hint_usage_rate']
        insights.append(f"💡 <strong>Hint Usage:</strong> {hint_rate:.1f}% of students use hints - {'high engagement' if hint_rate > 50 else 'low adoption'}")
        
        # Response time insight
        avg_time = eda_summary['avg_response_time']
        if avg_time < 200:
            insights.append(f"⚡ <strong>Quick Responses:</strong> Average response time is {avg_time:.0f}s - students work efficiently")
        else:
            insights.append(f"🕐 <strong>Longer Responses:</strong> Average response time is {avg_time:.0f}s - complex problems or struggling students")
        
        # Confidence insight
        avg_confidence = eda_summary['avg_confidence']
        insights.append(f"💪 <strong>Student Confidence:</strong> Average rating of {avg_confidence:.1f}/5 - {'high' if avg_confidence > 3.5 else 'moderate' if avg_confidence > 2.5 else 'low'} confidence levels")
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting User Interactions Analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Generate EDA summary
        print("📊 Generating EDA summary...")
        eda_summary = self.generate_eda_summary()
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        visuals = {
            'study_hours_accuracy': self.create_study_hours_accuracy_scatter(),
            'time_day_heatmap': self.create_time_of_day_performance_heatmap(),
            'difficulty_response': self.create_difficulty_response_time_violin(),
            'hint_impact': self.create_hint_usage_impact_comparison(),
            'grade_topic_performance': self.create_grade_level_topic_performance(),
            'confidence_accuracy': self.create_confidence_accuracy_relationship()
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
                'total_interactions': eda_summary['total_interactions'],
                'unique_users': eda_summary['unique_users'],
                'avg_accuracy': eda_summary['avg_accuracy'],
                'avg_response_time': eda_summary['avg_response_time'],
                'hint_usage_rate': eda_summary['hint_usage_rate']
            },
            'eda': {
                'topics_distribution': str(eda_summary['topics_distribution']),
                'grade_level_distribution': str(eda_summary['grade_level_distribution']),
                'time_of_day_distribution': str(eda_summary['time_of_day_distribution']),
                'avg_confidence': eda_summary['avg_confidence']
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
    visualizer = UserInteractionsVisualizer('static/datasets/user_interactions.csv')
    
    # Run complete analysis
    results = visualizer.run_complete_analysis()
    
    if results:
        print(f"📊 Generated {len(results['visuals'])} visualizations")
        print(f"📈 EDA Summary: {results['overview']}")
    else:
        print("❌ Analysis failed!")
