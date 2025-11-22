import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProblemBankVisualizer:
    def __init__(self, data_path):
        """
        Initialize the visualizer with problem bank data
        """
        self.data_path = data_path
        self.df = None
        self.output_dir = 'static/images'
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the problem bank data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {len(self.df)} records")
            
            # Basic data cleaning
            self.df['last_updated'] = pd.to_datetime(self.df['last_updated'])
            self.df['success_rate'] = pd.to_numeric(self.df['success_rate'], errors='coerce')
            self.df['avg_solving_time_seconds'] = pd.to_numeric(self.df['avg_solving_time_seconds'], errors='coerce')
            self.df['equation_complexity_score'] = pd.to_numeric(self.df['equation_complexity_score'], errors='coerce')
            self.df['avg_attempts_to_solve'] = pd.to_numeric(self.df['avg_attempts_to_solve'], errors='coerce')
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        eda_summary = {
            'total_problems': len(self.df),
            'unique_topics': self.df['topic'].nunique(),
            'date_range': {
                'start': self.df['last_updated'].min().strftime('%Y-%m-%d'),
                'end': self.df['last_updated'].max().strftime('%Y-%m-%d')
            },
            'avg_success_rate': round(self.df['success_rate'].mean() * 100, 2),
            'avg_solving_time': round(self.df['avg_solving_time_seconds'].mean(), 2),
            'avg_complexity': round(self.df['equation_complexity_score'].mean(), 2),
            'topics_distribution': self.df['topic'].value_counts().to_dict(),
            'difficulty_distribution': self.df['difficulty_level'].value_counts().to_dict(),
            'source_distribution': self.df['source'].value_counts().to_dict(),
            'verified_percentage': round((self.df['verified_by_human'].sum() / len(self.df)) * 100, 2)
        }
        
        return eda_summary
    
    def create_complexity_vs_success_scatter(self):
        """Visualization 1: Equation Complexity vs Success Rate"""
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot with color by difficulty
        colors = {1: '#96CEB4', 2: '#4ECDC4', 3: '#45B7D1', 4: '#FFA07A', 5: '#FF6B6B'}
        
        for difficulty in sorted(self.df['difficulty_level'].unique()):
            data = self.df[self.df['difficulty_level'] == difficulty]
            plt.scatter(data['equation_complexity_score'], 
                       data['success_rate'],
                       alpha=0.7,
                       s=100,
                       c=colors.get(difficulty, '#888888'),
                       label=f'Difficulty {difficulty}',
                       edgecolors='white',
                       linewidth=0.5)
        
        plt.xlabel('Equation Complexity Score', fontsize=12, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
        plt.title('Equation Complexity vs Success Rate: Does Complexity Predict Difficulty?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(title='Difficulty Level', frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add trend line
        valid_data = self.df.dropna(subset=['equation_complexity_score', 'success_rate'])
        if len(valid_data) > 0:
            z = np.polyfit(valid_data['equation_complexity_score'], 
                          valid_data['success_rate'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['equation_complexity_score'].min(), 
                                valid_data['equation_complexity_score'].max(), 100)
            plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/complexity_success_scatter.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_difficulty_time_boxplot(self):
        """Visualization 2: Difficulty Level vs Solving Time"""
        plt.figure(figsize=(12, 6))
        
        # Filter out extreme outliers for better visualization
        Q1 = self.df['avg_solving_time_seconds'].quantile(0.25)
        Q3 = self.df['avg_solving_time_seconds'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = self.df[
            (self.df['avg_solving_time_seconds'] >= Q1 - 1.5 * IQR) & 
            (self.df['avg_solving_time_seconds'] <= Q3 + 1.5 * IQR)
        ]
        
        # Create boxplot
        difficulty_groups = [filtered_df[filtered_df['difficulty_level'] == i]['avg_solving_time_seconds'].dropna() 
                            for i in sorted(filtered_df['difficulty_level'].unique())]
        
        bp = plt.boxplot(difficulty_groups, 
                        labels=sorted(filtered_df['difficulty_level'].unique()),
                        patch_artist=True,
                        notch=True,
                        showmeans=True)
        
        # Color the boxes
        colors = ['#96CEB4', '#4ECDC4', '#45B7D1', '#FFA07A', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        plt.ylabel('Average Solving Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Difficulty Level vs Solving Time: Time Distribution & Outliers', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add mean values as text
        for i, group in enumerate(difficulty_groups):
            if len(group) > 0:
                mean_val = np.mean(group)
                plt.text(i+1, mean_val, f'{mean_val:.0f}s', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/difficulty_time_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_topic_success_stacked_bar(self):
        """Visualization 3: Topic vs Success Rate (Stacked by Difficulty)"""
        plt.figure(figsize=(14, 7))
        
        # Pivot data for stacked bar chart
        pivot_data = self.df.pivot_table(
            values='success_rate',
            index='topic',
            columns='difficulty_level',
            aggfunc='mean'
        )
        
        # Create stacked bar chart
        pivot_data.plot(kind='bar', stacked=False, 
                       color=['#96CEB4', '#4ECDC4', '#45B7D1', '#FFA07A', '#FF6B6B'],
                       width=0.8, edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Topic', fontsize=12, fontweight='bold')
        plt.ylabel('Average Success Rate', fontsize=12, fontweight='bold')
        plt.title('Topic Performance by Difficulty Level: Which Topics Are Toughest?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(title='Difficulty', labels=[f'Level {i}' for i in range(1, 6)])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/topic_success_stacked.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_success_rate_histogram(self):
        """Visualization 4: Success Rate Distribution"""
        plt.figure(figsize=(12, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(self.df['success_rate'].dropna(), 
                                     bins=25, 
                                     color='#45B7D1', 
                                     alpha=0.7, 
                                     edgecolor='black', 
                                     linewidth=1.2)
        
        # Color gradient
        cm = plt.cm.RdYlGn
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers
        col = (col - min(col)) / (max(col) - min(col))
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        # Add mean line
        mean_success = self.df['success_rate'].mean()
        plt.axvline(mean_success, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_success:.2f}')
        
        # Add median line
        median_success = self.df['success_rate'].median()
        plt.axvline(median_success, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_success:.2f}')
        
        plt.xlabel('Success Rate', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency (Number of Problems)', fontsize=12, fontweight='bold')
        plt.title('Overall Problem Difficulty Distribution: Success Rate Histogram', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/success_rate_histogram.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_additional_visualizations(self):
        """Create 2 additional supporting visualizations"""
        
        # Viz 5: Grade Level Distribution
        plt.figure(figsize=(10, 6))
        grade_counts = self.df['grade_level'].value_counts().sort_index()
        bars = plt.bar(grade_counts.index, grade_counts.values, 
                      color='#4ECDC4', edgecolor='black', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Grade Level', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Problems', fontsize=12, fontweight='bold')
        plt.title('Problem Distribution by Grade Level', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        
        filename_grade = f"{self.output_dir}/grade_distribution.png"
        plt.savefig(filename_grade, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Viz 6: Complexity Distribution
        plt.figure(figsize=(10, 6))
        complexity_counts = self.df['equation_complexity_score'].value_counts().sort_index()
        bars = plt.bar(complexity_counts.index, complexity_counts.values,
                      color='#45B7D1', edgecolor='black', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Equation Complexity Score', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Problems', fontsize=12, fontweight='bold')
        plt.title('Distribution of Problem Complexity', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        
        filename_complex = f"{self.output_dir}/complexity_distribution.png"
        plt.savefig(filename_complex, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return (filename_grade.replace('static/', ''),
                filename_complex.replace('static/', ''))
    
    def perform_anova_test(self):
        """Hypothesis Test: ANOVA - Does difficulty level affect solving time?"""
        
        # Group data by difficulty level
        groups = []
        difficulty_levels = sorted(self.df['difficulty_level'].unique())
        
        for level in difficulty_levels:
            group_data = self.df[self.df['difficulty_level'] == level]['avg_solving_time_seconds'].dropna()
            groups.append(group_data)
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        # Filter outliers for better visualization
        Q1 = self.df['avg_solving_time_seconds'].quantile(0.25)
        Q3 = self.df['avg_solving_time_seconds'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = self.df[
            (self.df['avg_solving_time_seconds'] >= Q1 - 1.5 * IQR) & 
            (self.df['avg_solving_time_seconds'] <= Q3 + 1.5 * IQR)
        ]
        
        # Boxplot with violin overlay
        positions = range(1, len(difficulty_levels) + 1)
        bp = plt.boxplot([filtered_df[filtered_df['difficulty_level'] == level]['avg_solving_time_seconds'].dropna() 
                          for level in difficulty_levels],
                         labels=difficulty_levels,
                         positions=positions,
                         patch_artist=True,
                         notch=True,
                         showmeans=True)
        
        # Color the boxes
        colors = ['#96CEB4', '#4ECDC4', '#45B7D1', '#FFA07A', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors[:len(difficulty_levels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        plt.ylabel('Average Solving Time (seconds)', fontsize=12, fontweight='bold')
        plt.title(f'ANOVA Test: Difficulty Level vs Solving Time\nF-statistic = {f_stat:.4f}, p-value = {p_value:.6f}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add significance annotation
        if p_value < 0.05:
            plt.text(0.5, 0.95, '✓ Statistically Significant (p < 0.05)', 
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    fontsize=11, fontweight='bold', ha='center')
        else:
            plt.text(0.5, 0.95, '✗ Not Statistically Significant (p ≥ 0.05)',
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                    fontsize=11, fontweight='bold', ha='center')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/anova_difficulty_time.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Calculate effect size (eta-squared)
        grand_mean = self.df['avg_solving_time_seconds'].mean()
        ss_between = sum([len(self.df[self.df['difficulty_level'] == level]) * 
                         (self.df[self.df['difficulty_level'] == level]['avg_solving_time_seconds'].mean() - grand_mean)**2 
                         for level in difficulty_levels])
        ss_total = sum((self.df['avg_solving_time_seconds'].dropna() - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        hypothesis_results = {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_value, 6),
            'eta_squared': round(eta_squared, 4),
            'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
            'null_hypothesis': 'All difficulty levels have the same mean solving time',
            'alternative_hypothesis': 'At least one difficulty level has a different mean solving time',
            'conclusion': 'Reject H₀ - Difficulty levels significantly affect solving time' if p_value < 0.05 
                         else 'Fail to reject H₀ - No significant difference in solving time across difficulty levels'
        }
        
        return hypothesis_results, filename.replace('static/', '')
    
    def perform_multiple_regression(self):
        """Predictive Model: Multiple Linear Regression for solving time"""
        
        # Prepare features
        df_model = self.df.dropna(subset=['avg_solving_time_seconds', 'difficulty_level', 
                                          'equation_complexity_score', 'problem_length_words'])
        
        # Feature matrix
        X = df_model[['difficulty_level', 'equation_complexity_score', 
                     'problem_length_words', 'equation_count']]
        y = df_model['avg_solving_time_seconds']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Model evaluation
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Create visualization 1 - Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.6, color='#4ECDC4', edgecolors='black', linewidth=0.5, s=80)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Solving Time (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Solving Time (seconds)', fontsize=12, fontweight='bold')
        plt.title(f'Multiple Linear Regression: Actual vs Predicted\nR² = {r2_test:.4f}, RMSE = {rmse:.2f}s',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        filename_pred = f"{self.output_dir}/regression_actual_vs_predicted.png"
        plt.savefig(filename_pred, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create visualization 2 - Feature Coefficients
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in feature_importance['coefficient']]
        bars = plt.barh(feature_importance['feature'], feature_importance['coefficient'],
                color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, feature_importance['coefficient'])):
            plt.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                    ha='left' if val > 0 else 'right', va='center', fontweight='bold', fontsize=10)
        
        plt.xlabel('Coefficient Value (seconds per unit)', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title('Feature Impact on Solving Time: Regression Coefficients', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        filename_coef = f"{self.output_dir}/regression_coefficients.png"
        plt.savefig(filename_coef, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Results dictionary
        regression_results = {
            'r2_train': round(r2_train, 4),
            'r2_test': round(r2_test, 4),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'intercept': round(model.intercept_, 2),
            'top_feature': feature_importance.iloc[0]['feature'],
            'top_coefficient': round(feature_importance.iloc[0]['coefficient'], 4),
            'equation': f"Time = {model.intercept_:.2f} + {model.coef_[0]:.2f}×Difficulty + {model.coef_[1]:.2f}×Complexity + ...",
            'interpretation': f"Model predicts solving time with R² = {r2_test:.3f}. "
                            f"Each unit increase in {feature_importance.iloc[0]['feature']} adds ~{abs(feature_importance.iloc[0]['coefficient']):.1f} seconds."
        }
        
        return regression_results, filename_pred.replace('static/', ''), filename_coef.replace('static/', '')
    
    def generate_insights(self, eda_summary, hypothesis_results, regression_results):
        """Generate automated insights from the analysis"""
        insights = []
        
        # Success rate insight
        avg_success = eda_summary['avg_success_rate']
        if avg_success > 70:
            insights.append(f"🎯 <strong>High Success Rate:</strong> Problems have {avg_success:.1f}% average success rate")
        elif avg_success > 50:
            insights.append(f"⚠️ <strong>Moderate Success Rate:</strong> Problems have {avg_success:.1f}% average success rate")
        else:
            insights.append(f"🚨 <strong>Low Success Rate:</strong> Problems are challenging with {avg_success:.1f}% success rate")
        
        # Solving time insight
        avg_time = eda_summary['avg_solving_time']
        insights.append(f"⏱️ <strong>Average Solving Time:</strong> Students take {avg_time:.0f} seconds per problem")
        
        # Complexity insight
        avg_complexity = eda_summary['avg_complexity']
        insights.append(f"🔢 <strong>Problem Complexity:</strong> Average equation complexity score is {avg_complexity:.1f}/10")
        
        # Hypothesis test insight
        if hypothesis_results['p_value'] < 0.05:
            insights.append(f"🔬 <strong>Difficulty Impact Confirmed:</strong> Significant difference found (p = {hypothesis_results['p_value']:.4f})")
        else:
            insights.append(f"📊 <strong>Difficulty Impact Unclear:</strong> No significant difference (p = {hypothesis_results['p_value']:.4f})")
        
        # Predictive model insight
        model_r2 = regression_results['r2_test']
        insights.append(f"🤖 <strong>Predictive Model:</strong> Can predict solving time with R² = {model_r2:.3f}")
        
        # Verification insight
        verified_pct = eda_summary['verified_percentage']
        insights.append(f"✓ <strong>Quality Assurance:</strong> {verified_pct:.1f}% of problems are human-verified")
        
        return insights
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Problem Bank Analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Generate EDA summary
        print("📊 Generating EDA summary...")
        eda_summary = self.generate_eda_summary()
        
        # Create main visualizations
        print("🎨 Creating visualizations...")
        visuals = {
            'complexity_success': self.create_complexity_vs_success_scatter(),
            'difficulty_time': self.create_difficulty_time_boxplot(),
            'topic_success': self.create_topic_success_stacked_bar(),
            'success_histogram': self.create_success_rate_histogram()
        }
        
        # Additional visualizations
        print("🎨 Creating additional visualizations...")
        grade_plot, complex_plot = self.create_additional_visualizations()
        visuals['grade_distribution'] = grade_plot
        visuals['complexity_distribution'] = complex_plot
        
        # Perform statistical analyses
        print("🔬 Performing hypothesis testing...")
        hypothesis_results, hypothesis_plot = self.perform_anova_test()
        visuals['hypothesis_plot'] = hypothesis_plot
        
        print("📈 Performing predictive modeling...")
        regression_results, pred_plot, coef_plot = self.perform_multiple_regression()
        visuals['regression_actual_predicted'] = pred_plot
        visuals['regression_coefficients'] = coef_plot
        
        # Generate insights
        print("💡 Generating insights...")
        insights = self.generate_insights(eda_summary, hypothesis_results, regression_results)
        
        # Compile final results
        self.results = {
            'overview': {
                'total_problems': eda_summary['total_problems'],
                'unique_topics': eda_summary['unique_topics'],
                'avg_success_rate': eda_summary['avg_success_rate'],
                'avg_solving_time': eda_summary['avg_solving_time'],
                'avg_complexity': eda_summary['avg_complexity'],
                'verified_percentage': eda_summary['verified_percentage']
            },
            'eda': {
                'topics_distribution': str(eda_summary['topics_distribution']),
                'difficulty_distribution': str(eda_summary['difficulty_distribution']),
                'source_distribution': str(eda_summary['source_distribution']),
                'date_range': eda_summary['date_range']
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
    visualizer = ProblemBankVisualizer('static/datasets/problem_bank.csv')
    
    # Run complete analysis
    results = visualizer.run_complete_analysis()
    
    if results:
        print(f"📈 Generated {len(results['visuals'])} visualizations")
        print(f"📊 EDA Summary: {results['overview']}")
    else:
        print("❌ Analysis failed!")
