import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
            self.df['study_hours_per_week'] = pd.to_numeric(self.df['study_hours_per_week'], errors='coerce')
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary"""
        eda_summary = {
            'total_records': len(self.df),
            'unique_users': self.df['user_id'].nunique(),
            'date_range': {
                'start': self.df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': self.df['timestamp'].max().strftime('%Y-%m-%d')
            },
            'avg_accuracy': round(self.df['user_accuracy'].mean() * 100, 2),
            'avg_response_time': round(self.df['response_time_seconds'].mean(), 2),
            'avg_study_hours': round(self.df['study_hours_per_week'].mean(), 2),
            'learning_styles': self.df['learning_style'].value_counts().to_dict(),
            'topics_distribution': self.df['topic'].value_counts().to_dict(),
            'device_types': self.df['device_type'].value_counts().to_dict(),
            'hint_usage_rate': round((self.df['hint_used'].value_counts().get('yes', 0) / len(self.df)) * 100, 2)
        }
        
        return eda_summary
    
    def create_study_hours_vs_accuracy_scatter(self):
        """Visualization 1: Study Hours vs Accuracy Scatter Plot"""
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot with color by difficulty
        colors = {'1': '#96CEB4', '2': '#4ECDC4', '3': '#45B7D1', '4': '#FF6B6B'}
        
        for difficulty in sorted(self.df['difficulty_level'].unique()):
            data = self.df[self.df['difficulty_level'] == difficulty]
            plt.scatter(data['study_hours_per_week'], 
                       data['user_accuracy'],
                       alpha=0.6,
                       s=80,
                       c=colors.get(str(difficulty), '#888888'),
                       label=f'Difficulty {difficulty}',
                       edgecolors='white',
                       linewidth=0.5)
        
        plt.xlabel('Study Hours per Week', fontsize=12, fontweight='bold')
        plt.ylabel('User Accuracy', fontsize=12, fontweight='bold')
        plt.title('Study Hours vs Accuracy: Does More Studying Lead to Better Performance?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(title='Difficulty Level', frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add trend line
        z = np.polyfit(self.df['study_hours_per_week'].dropna(), 
                      self.df.loc[self.df['study_hours_per_week'].notna(), 'user_accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['study_hours_per_week'].sort_values(), 
                p(self.df['study_hours_per_week'].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/study_hours_accuracy.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_learning_style_boxplot(self):
        """Visualization 2: Learning Style Performance Comparison"""
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        learning_styles = self.df.groupby('learning_style')['user_accuracy'].apply(list)
        
        bp = plt.boxplot(learning_styles.values, 
                        labels=learning_styles.index,
                        patch_artist=True,
                        notch=True,
                        showmeans=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Learning Style', fontsize=12, fontweight='bold')
        plt.ylabel('User Accuracy', fontsize=12, fontweight='bold')
        plt.title('Learning Style vs Performance: Which Style Performs Best?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add mean values as text
        for i, style in enumerate(learning_styles.index):
            mean_val = np.mean(learning_styles[style])
            plt.text(i+1, mean_val, f'{mean_val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/learning_style_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_day_time_heatmap(self):
        """Visualization 3: Day/Time Performance Heatmap"""
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        heatmap_data = self.df.pivot_table(
            values='user_accuracy',
            index='time_of_day',
            columns='day_of_week',
            aggfunc='mean'
        )
        
        # Order time of day
        time_order = ['morning', 'afternoon', 'evening', 'night']
        heatmap_data = heatmap_data.reindex(time_order)
        
        # Order days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data[day_order]
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Average Accuracy'},
                   linewidths=1,
                   linecolor='white',
                   vmin=0,
                   vmax=1)
        
        plt.title('Best Performance Times: Day & Time Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
        plt.ylabel('Time of Day', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/day_time_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_student_progress_line(self):
        """Visualization 4: Student Progress Over Attempts"""
        plt.figure(figsize=(12, 6))
        
        # Calculate average accuracy by attempt number
        progress_data = self.df.groupby('attempt_number')['user_accuracy'].agg(['mean', 'std', 'count'])
        
        # Plot line with confidence interval
        plt.plot(progress_data.index, progress_data['mean'], 
                marker='o', linewidth=3, markersize=10, 
                color='#4ECDC4', label='Average Accuracy')
        
        # Add confidence interval
        plt.fill_between(progress_data.index,
                        progress_data['mean'] - progress_data['std'],
                        progress_data['mean'] + progress_data['std'],
                        alpha=0.3, color='#4ECDC4')
        
        # Add data point counts
        for idx, row in progress_data.iterrows():
            plt.text(idx, row['mean'], f"n={row['count']}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.xlabel('Attempt Number', fontsize=12, fontweight='bold')
        plt.ylabel('Average Accuracy', fontsize=12, fontweight='bold')
        plt.title('Student Learning Progress: How Performance Improves with Attempts', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(frameon=True, shadow=True)
        
        plt.tight_layout()
        filename = f"{self.output_dir}/student_progress_line.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_response_time_histogram(self):
        """Visualization 5: Response Time Distribution"""
        plt.figure(figsize=(12, 6))
        
        # Remove extreme outliers for better visualization
        Q1 = self.df['response_time_seconds'].quantile(0.25)
        Q3 = self.df['response_time_seconds'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = self.df[
            (self.df['response_time_seconds'] >= Q1 - 1.5 * IQR) & 
            (self.df['response_time_seconds'] <= Q3 + 1.5 * IQR)
        ]['response_time_seconds']
        
        # Create histogram
        n, bins, patches = plt.hist(filtered_data, bins=30, 
                                     color='#45B7D1', alpha=0.7, 
                                     edgecolor='black', linewidth=1.2)
        
        # Color gradient
        cm = plt.cm.viridis
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        
        # Add mean line
        mean_time = filtered_data.mean()
        plt.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_time:.1f}s')
        
        # Add median line
        median_time = filtered_data.median()
        plt.axvline(median_time, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_time:.1f}s')
        
        plt.xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Student Response Time Distribution: How Long Do Students Take?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        filename = f"{self.output_dir}/response_time_histogram.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename.replace('static/', '')
    
    def create_additional_visualizations(self):
        """Create 3 additional supporting visualizations"""
        
        # Viz 6: Difficulty vs Success Rate
        plt.figure(figsize=(10, 6))
        difficulty_success = self.df.groupby('difficulty_level')['user_accuracy'].mean() * 100
        bars = plt.bar(difficulty_success.index, difficulty_success.values, 
                      color=['#96CEB4', '#4ECDC4', '#45B7D1', '#FF6B6B'],
                      edgecolor='black', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        plt.ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Difficulty Impact on Success Rate', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        
        filename_diff = f"{self.output_dir}/difficulty_success.png"
        plt.savefig(filename_diff, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Viz 7: Hint Usage Impact
        plt.figure(figsize=(10, 6))
        hint_impact = self.df.groupby('hint_used')['user_accuracy'].mean() * 100
        colors_hint = {'yes': '#4ECDC4', 'no': '#FF6B6B'}
        bars = plt.bar(hint_impact.index, hint_impact.values,
                      color=[colors_hint.get(x, '#888888') for x in hint_impact.index],
                      edgecolor='black', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.xlabel('Hint Used', fontsize=12, fontweight='bold')
        plt.ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Impact of Hint Usage on Performance', fontsize=14, fontweight='bold', pad=20)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        
        filename_hint = f"{self.output_dir}/hint_usage_impact.png"
        plt.savefig(filename_hint, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Viz 8: Grade Level Performance
        plt.figure(figsize=(12, 6))
        grade_performance = self.df.groupby('grade_level')['user_accuracy'].agg(['mean', 'count'])
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for accuracy
        ax1.bar(grade_performance.index, grade_performance['mean'] * 100,
               color='#45B7D1', alpha=0.7, label='Avg Accuracy (%)')
        ax1.set_xlabel('Grade Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold', color='#45B7D1')
        ax1.tick_params(axis='y', labelcolor='#45B7D1')
        
        # Line chart for count
        ax2 = ax1.twinx()
        ax2.plot(grade_performance.index, grade_performance['count'],
                color='red', marker='o', linewidth=2, markersize=8, label='Sample Count')
        ax2.set_ylabel('Number of Students', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Performance by Grade Level', fontsize=14, fontweight='bold', pad=20)
        fig.tight_layout()
        
        filename_grade = f"{self.output_dir}/grade_performance.png"
        plt.savefig(filename_grade, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return (filename_diff.replace('static/', ''), 
                filename_hint.replace('static/', ''),
                filename_grade.replace('static/', ''))
    
    def perform_anova_test(self):
        """Hypothesis Test: ANOVA - Do learning styles affect accuracy?"""
        
        # Group data by learning style
        groups = []
        learning_styles = self.df['learning_style'].unique()
        
        for style in learning_styles:
            group_data = self.df[self.df['learning_style'] == style]['user_accuracy'].dropna()
            groups.append(group_data)
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Create visualization
        plt.figure(figsize=(12, 7))
        
        # Boxplot with violin overlay
        positions = range(1, len(learning_styles) + 1)
        bp = plt.boxplot([self.df[self.df['learning_style'] == style]['user_accuracy'].dropna() 
                          for style in learning_styles],
                         labels=learning_styles,
                         positions=positions,
                         patch_artist=True,
                         notch=True,
                         showmeans=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for patch, color in zip(bp['boxes'], colors[:len(learning_styles)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.xlabel('Learning Style', fontsize=12, fontweight='bold')
        plt.ylabel('User Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'ANOVA Test: Learning Style vs Accuracy\nF-statistic = {f_stat:.4f}, p-value = {p_value:.6f}',
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
        filename = f"{self.output_dir}/anova_learning_style.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Calculate effect size (eta-squared)
        grand_mean = self.df['user_accuracy'].mean()
        ss_between = sum([len(self.df[self.df['learning_style'] == style]) * 
                         (self.df[self.df['learning_style'] == style]['user_accuracy'].mean() - grand_mean)**2 
                         for style in learning_styles])
        ss_total = sum((self.df['user_accuracy'] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        hypothesis_results = {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_value, 6),
            'eta_squared': round(eta_squared, 4),
            'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
            'null_hypothesis': 'All learning styles have the same mean accuracy',
            'alternative_hypothesis': 'At least one learning style has a different mean accuracy',
            'conclusion': 'Reject H₀ - Learning styles significantly affect accuracy' if p_value < 0.05 
                         else 'Fail to reject H₀ - No significant difference between learning styles'
        }
        
        return hypothesis_results, filename.replace('static/', '')
    
    def perform_logistic_regression(self):
        """Predictive Model: Logistic Regression for problem-solving success"""
        
        # Prepare features
        df_model = self.df.dropna(subset=['user_accuracy', 'difficulty_level', 
                                          'study_hours_per_week', 'learning_style'])
        
        # Binary classification: 1 = success, 0 = failure
        df_model['success'] = (df_model['user_accuracy'] == 1).astype(int)
        
        # One-hot encode learning style
        learning_style_dummies = pd.get_dummies(df_model['learning_style'], prefix='style')
        
        # Prepare feature matrix
        X = pd.concat([
            df_model[['difficulty_level', 'study_hours_per_week', 'previous_math_score']],
            learning_style_dummies
        ], axis=1)
        
        y = df_model['success']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Model evaluation
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Create visualization - Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Failure', 'Success'],
                   yticklabels=['Failure', 'Success'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Logistic Regression: Confusion Matrix\nAccuracy = {accuracy:.2%}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename_cm = f"{self.output_dir}/logistic_confusion_matrix.png"
        plt.savefig(filename_cm, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Feature Importance Plot
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in feature_importance['coefficient']]
        plt.barh(feature_importance['feature'], feature_importance['coefficient'],
                color=colors, edgecolor='black', linewidth=1)
        
        plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title('Feature Importance: What Predicts Success?', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        filename_fi = f"{self.output_dir}/logistic_feature_importance.png"
        plt.savefig(filename_fi, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Results dictionary
        regression_results = {
            'accuracy': round(accuracy * 100, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'true_positives': int(conf_matrix[1, 1]),
            'true_negatives': int(conf_matrix[0, 0]),
            'false_positives': int(conf_matrix[0, 1]),
            'false_negatives': int(conf_matrix[1, 0]),
            'top_positive_feature': feature_importance.iloc[0]['feature'],
            'top_positive_coefficient': round(feature_importance.iloc[0]['coefficient'], 4),
            'interpretation': f"Model predicts success with {accuracy:.1%} accuracy. "
                            f"Key factors: difficulty level, study hours, and learning style."
        }
        
        return regression_results, filename_cm.replace('static/', ''), filename_fi.replace('static/', '')
    
    def generate_insights(self, eda_summary, hypothesis_results, regression_results):
        """Generate automated insights from the analysis"""
        insights = []
        
        # Accuracy insight
        avg_accuracy = eda_summary['avg_accuracy']
        if avg_accuracy > 75:
            insights.append(f"🎯 <strong>High Performance:</strong> Students achieve {avg_accuracy:.1f}% average accuracy")
        elif avg_accuracy > 50:
            insights.append(f"⚠️ <strong>Moderate Performance:</strong> Students achieve {avg_accuracy:.1f}% average accuracy")
        else:
            insights.append(f"🚨 <strong>Low Performance:</strong> Students achieve {avg_accuracy:.1f}% average accuracy")
        
        # Study hours insight
        avg_study = eda_summary['avg_study_hours']
        insights.append(f"📚 <strong>Study Habits:</strong> Students study an average of {avg_study:.1f} hours per week")
        
        # Hypothesis test insight
        if hypothesis_results['p_value'] < 0.05:
            insights.append(f"🔬 <strong>Learning Styles Matter:</strong> Significant difference found (p = {hypothesis_results['p_value']:.4f})")
        else:
            insights.append(f"📊 <strong>Learning Styles Similar:</strong> No significant difference (p = {hypothesis_results['p_value']:.4f})")
        
        # Predictive model insight
        model_accuracy = regression_results['accuracy']
        insights.append(f"🤖 <strong>Predictive Model:</strong> Can predict success with {model_accuracy:.1f}% accuracy")
        
        # Response time insight
        avg_time = eda_summary['avg_response_time']
        insights.append(f"⏱️ <strong>Average Response Time:</strong> Students take {avg_time:.0f} seconds per problem")
        
        # Hint usage insight
        hint_rate = eda_summary['hint_usage_rate']
        insights.append(f"💡 <strong>Hint Usage:</strong> {hint_rate:.1f}% of attempts use hints")
        
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
        
        # Create main visualizations
        print("🎨 Creating visualizations...")
        visuals = {
            'study_hours_accuracy': self.create_study_hours_vs_accuracy_scatter(),
            'learning_style_boxplot': self.create_learning_style_boxplot(),
            'day_time_heatmap': self.create_day_time_heatmap(),
            'student_progress_line': self.create_student_progress_line(),
            'response_time_histogram': self.create_response_time_histogram()
        }
        
        # Additional visualizations
        print("🎨 Creating additional visualizations...")
        diff_plot, hint_plot, grade_plot = self.create_additional_visualizations()
        visuals['difficulty_success'] = diff_plot
        visuals['hint_usage'] = hint_plot
        visuals['grade_performance'] = grade_plot
        
        # Perform statistical analyses
        print("🔬 Performing hypothesis testing...")
        hypothesis_results, hypothesis_plot = self.perform_anova_test()
        visuals['hypothesis_plot'] = hypothesis_plot
        
        print("🤖 Performing predictive modeling...")
        regression_results, cm_plot, fi_plot = self.perform_logistic_regression()
        visuals['confusion_matrix'] = cm_plot
        visuals['feature_importance'] = fi_plot
        
        # Generate insights
        print("💡 Generating insights...")
        insights = self.generate_insights(eda_summary, hypothesis_results, regression_results)
        
        # Compile final results
        self.results = {
            'overview': {
                'total_records': eda_summary['total_records'],
                'unique_users': eda_summary['unique_users'],
                'avg_accuracy': eda_summary['avg_accuracy'],
                'avg_response_time': eda_summary['avg_response_time'],
                'avg_study_hours': eda_summary['avg_study_hours'],
                'hint_usage_rate': eda_summary['hint_usage_rate']
            },
            'eda': {
                'learning_styles': str(eda_summary['learning_styles']),
                'topics_distribution': str(eda_summary['topics_distribution']),
                'device_types': str(eda_summary['device_types']),
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
    visualizer = UserInteractionsVisualizer('static/datasets/user_interactions.csv')
    
    # Run complete analysis
    results = visualizer.run_complete_analysis()
    
    if results:
        print(f"📈 Generated {len(results['visuals'])} visualizations")
        print(f"📊 EDA Summary: {results['overview']}")
    else:
        print("❌ Analysis failed!")