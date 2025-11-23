# visualizations/gsm8k_viz.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class GSM8KVisualizer:
    def __init__(self):
        """
        Initialize GSM8K dataset visualizer
        """
        self.ds = None
        self.train_df = None
        self.validation_df = None
        self.combined_df = None
        self.output_dir = 'static/images/gsm8k'
        self.results = {}
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_dataset(self):
        """Load GSM8K dataset from Hugging Face"""
        try:
            from datasets import load_dataset
            self.ds = load_dataset("openai/gsm8k", "main")
            
            # Convert to pandas DataFrames
            self.train_df = pd.DataFrame(self.ds['train'])
            self.validation_df = pd.DataFrame(self.ds['test'])  # Note: GSM8K uses 'test' as validation
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   Train: {len(self.train_df)} samples")
            print(f"   Validation: {len(self.validation_df)} samples")
            
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def extract_features(self):
        """Extract features from GSM8K dataset"""
        print("🔍 Extracting features from GSM8K dataset...")
        
        try:
            # Combine datasets for analysis FIRST
            train_with_split = self.train_df.copy()
            val_with_split = self.validation_df.copy()
            train_with_split['split'] = 'train'
            val_with_split['split'] = 'validation'
            self.combined_df = pd.concat([train_with_split, val_with_split])
            
            # Text length features
            self.combined_df['question_length'] = self.combined_df['question'].apply(len)
            self.combined_df['question_word_count'] = self.combined_df['question'].apply(lambda x: len(x.split()))
            self.combined_df['answer_length'] = self.combined_df['answer'].apply(len)
            self.combined_df['answer_word_count'] = self.combined_df['answer'].apply(lambda x: len(x.split()))
            
            # Extract reasoning steps
            def count_reasoning_steps(answer_text):
                if isinstance(answer_text, str):
                    return len(re.findall(r'<<.*?>>', answer_text))
                return 0
            
            self.combined_df['reasoning_steps'] = self.combined_df['answer'].apply(count_reasoning_steps)
            
            # Extract final answer
            def extract_final_answer(answer_text):
                if isinstance(answer_text, str):
                    matches = re.findall(r'####\s*([-+]?\d*\.?\d+)', answer_text)
                    return matches[0] if matches else None
                return None
            
            self.combined_df['final_answer'] = self.combined_df['answer'].apply(extract_final_answer)
            
            # Convert final answer to numeric
            self.combined_df['final_answer_numeric'] = pd.to_numeric(
                self.combined_df['final_answer'], errors='coerce'
            )
            
            # Extract operations
            def detect_operation(text, patterns):
                if isinstance(text, str):
                    return any(re.search(pattern, text.lower()) for pattern in patterns)
                return False
            
            addition_patterns = [r'\bplus\b', r'\+', r'\badd\b', r'\bsum\b']
            subtraction_patterns = [r'\bminus\b', r'-', r'\bsubtract\b', r'\bdifference\b']
            multiplication_patterns = [r'\btimes\b', r'\bmultiplied\b', r'×', r'\*', r'\bproduct\b']
            division_patterns = [r'\bdivided\b', r'÷', r'/', r'\bquotient\b', r'\bper\b']
            
            self.combined_df['has_addition'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, addition_patterns)
            )
            self.combined_df['has_subtraction'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, subtraction_patterns)
            )
            self.combined_df['has_multiplication'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, multiplication_patterns)
            )
            self.combined_df['has_division'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, division_patterns)
            )
            
            # Count total operations
            self.combined_df['total_operations'] = (
                self.combined_df['has_addition'].astype(int) + 
                self.combined_df['has_subtraction'].astype(int) + 
                self.combined_df['has_multiplication'].astype(int) + 
                self.combined_df['has_division'].astype(int)
            )
            
            # Extract common entities
            money_patterns = [r'\$\d+', r'dollar', r'cent', r'price', r'cost']
            time_patterns = [r'\bhour', r'\bminute', r'\bsecond', r'\bday', r'\bweek', r'\bmonth', r'\byear']
            distance_patterns = [r'\bmile', r'\bkilometer', r'\bmeter', r'\bfoot', r'\binch']
            
            self.combined_df['has_money'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, money_patterns)
            )
            self.combined_df['has_time'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, time_patterns)
            )
            self.combined_df['has_distance'] = self.combined_df['question'].apply(
                lambda x: detect_operation(x, distance_patterns)
            )
            
            # Update train_df and validation_df with the new features
            self.train_df = self.combined_df[self.combined_df['split'] == 'train'].copy()
            self.validation_df = self.combined_df[self.combined_df['split'] == 'validation'].copy()
            
            print("✅ Feature extraction completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in feature extraction: {e}")
            return False
    
    def generate_dataset_summary(self):
        """Generate comprehensive dataset summary"""
        print("📊 Generating dataset summary...")
        
        summary = {
            'total_problems': len(self.combined_df),
            'train_problems': len(self.train_df),
            'validation_problems': len(self.validation_df),
            'avg_question_length': round(self.combined_df['question_word_count'].mean(), 1),
            'avg_answer_length': round(self.combined_df['answer_word_count'].mean(), 1),
            'avg_reasoning_steps': round(self.combined_df['reasoning_steps'].mean(), 1),
            'most_common_operation': self._get_most_common_operation(),
            'integer_answers_ratio': round((self.combined_df['final_answer_numeric'] % 1 == 0).mean() * 100, 1),
            'max_reasoning_steps': self.combined_df['reasoning_steps'].max(),
            'min_reasoning_steps': self.combined_df['reasoning_steps'].min()
        }
        
        # Operation frequencies
        operations = {
            'Addition': self.combined_df['has_addition'].mean() * 100,
            'Subtraction': self.combined_df['has_subtraction'].mean() * 100,
            'Multiplication': self.combined_df['has_multiplication'].mean() * 100,
            'Division': self.combined_df['has_division'].mean() * 100
        }
        summary['operation_frequencies'] = operations
        summary['most_frequent_operation'] = max(operations, key=operations.get)
        
        return summary
    
    def _get_most_common_operation(self):
        """Helper to find most common operation"""
        op_counts = {
            'addition': self.combined_df['has_addition'].sum(),
            'subtraction': self.combined_df['has_subtraction'].sum(),
            'multiplication': self.combined_df['has_multiplication'].sum(),
            'division': self.combined_df['has_division'].sum()
        }
        return max(op_counts, key=op_counts.get)
    
    def create_text_length_visualizations(self):
        """Create visualizations for text length analysis"""
        print("📏 Creating text length visualizations...")
        
        try:
            # 1. Question length histogram
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.combined_df['question_word_count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Question Word Count')
            plt.ylabel('Frequency')
            plt.title('Distribution of Question Lengths')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(self.combined_df['answer_word_count'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('Answer Word Count')
            plt.ylabel('Frequency')
            plt.title('Distribution of Answer Lengths')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/text_lengths_histogram.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Train vs Validation comparison
            plt.figure(figsize=(10, 6))
            
            train_data = self.combined_df[self.combined_df['split'] == 'train']
            val_data = self.combined_df[self.combined_df['split'] == 'validation']
            
            boxplot_data = [train_data['question_word_count'], val_data['question_word_count']]
            plt.boxplot(boxplot_data, labels=['Train', 'Validation'])
            plt.ylabel('Question Word Count')
            plt.title('Question Length: Train vs Validation')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/train_val_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Scatter plot: Question vs Answer length
            plt.figure(figsize=(10, 6))
            plt.scatter(self.combined_df['question_word_count'], 
                       self.combined_df['answer_word_count'], 
                       alpha=0.6, color='green', s=20)
            plt.xlabel('Question Word Count')
            plt.ylabel('Answer Word Count')
            plt.title('Question Length vs Answer Length')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = self.combined_df['question_word_count'].corr(self.combined_df['answer_word_count'])
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/question_answer_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'text_lengths_histogram': 'images/gsm8k/text_lengths_histogram.png',
                'train_val_comparison': 'images/gsm8k/train_val_comparison.png',
                'question_answer_scatter': 'images/gsm8k/question_answer_scatter.png'
            }
            
        except Exception as e:
            print(f"❌ Error in text length visualizations: {e}")
            return self._create_placeholder_visuals('text_length')
    
    def create_reasoning_analysis_visualizations(self):
        """Create visualizations for reasoning step analysis"""
        print("🔢 Creating reasoning analysis visualizations...")
        
        try:
            # 1. Reasoning steps distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            step_counts = self.combined_df['reasoning_steps'].value_counts().sort_index()
            
            plt.bar(step_counts.index, step_counts.values, color='lightseagreen', alpha=0.7)
            plt.xlabel('Number of Reasoning Steps')
            plt.ylabel('Frequency')
            plt.title('Distribution of Reasoning Steps')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(step_counts.values):
                plt.text(step_counts.index[i], v + 10, str(v), ha='center')
            
            plt.subplot(1, 2, 2)
            # Train vs Validation step comparison
            train_steps = self.train_df['reasoning_steps'].value_counts().sort_index()
            val_steps = self.validation_df['reasoning_steps'].value_counts().sort_index()
            
            x = np.arange(len(train_steps))
            width = 0.35
            
            plt.bar(x - width/2, train_steps.values, width, label='Train', alpha=0.7)
            plt.bar(x + width/2, val_steps.values, width, label='Validation', alpha=0.7)
            
            plt.xlabel('Reasoning Steps')
            plt.ylabel('Frequency')
            plt.title('Reasoning Steps: Train vs Validation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(x, train_steps.index)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/reasoning_steps_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'reasoning_steps_analysis': 'images/gsm8k/reasoning_steps_analysis.png'
            }
            
        except Exception as e:
            print(f"❌ Error in reasoning analysis: {e}")
            return self._create_placeholder_visuals('reasoning')
    
    def create_operation_analysis_visualizations(self):
        """Create visualizations for operation analysis"""
        print("➕ Creating operation analysis visualizations...")
        
        try:
            # 1. Operation frequency pie chart
            plt.figure(figsize=(10, 8))
            
            operations = ['Addition', 'Subtraction', 'Multiplication', 'Division']
            frequencies = [
                self.combined_df['has_addition'].mean() * 100,
                self.combined_df['has_subtraction'].mean() * 100,
                self.combined_df['has_multiplication'].mean() * 100,
                self.combined_df['has_division'].mean() * 100
            ]
            
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            plt.pie(frequencies, labels=operations, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Distribution of Arithmetic Operations')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/operations_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Multi-operation analysis
            plt.figure(figsize=(10, 6))
            
            multi_op_counts = self.combined_df['total_operations'].value_counts().sort_index()
            plt.bar(multi_op_counts.index, multi_op_counts.values, color='coral', alpha=0.7)
            plt.xlabel('Number of Different Operations')
            plt.ylabel('Frequency')
            plt.title('Problems with Multiple Operation Types')
            plt.grid(True, alpha=0.3)
            
            for i, v in enumerate(multi_op_counts.values):
                plt.text(multi_op_counts.index[i], v + 10, str(v), ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/multi_operations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'operations_pie': 'images/gsm8k/operations_pie.png',
                'multi_operations': 'images/gsm8k/multi_operations.png'
            }
            
        except Exception as e:
            print(f"❌ Error in operation analysis: {e}")
            return self._create_placeholder_visuals('operations')
    
    def create_topic_analysis_visualizations(self):
        """Create visualizations for topic analysis"""
        print("🏷️ Creating topic analysis visualizations...")
        
        try:
            # Topic inference based on keywords
            topics = {
                'Money': self.combined_df['has_money'].sum(),
                'Time': self.combined_df['has_time'].sum(),
                'Distance': self.combined_df['has_distance'].sum()
            }
            
            # 1. Topic distribution
            plt.figure(figsize=(10, 6))
            plt.bar(topics.keys(), topics.values(), color=['gold', 'lightblue', 'lightgreen'])
            plt.xlabel('Inferred Topics')
            plt.ylabel('Frequency')
            plt.title('Topic Distribution (Keyword-based Inference)')
            plt.grid(True, alpha=0.3)
            
            for i, v in enumerate(topics.values()):
                plt.text(i, v + 10, str(v), ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Word cloud for questions
            plt.figure(figsize=(12, 8))
            all_text = ' '.join(self.combined_df['question'].str.lower())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud: Most Common Words in Questions')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'topic_distribution': 'images/gsm8k/topic_distribution.png',
                'wordcloud': 'images/gsm8k/wordcloud.png'
            }
            
        except Exception as e:
            print(f"❌ Error in topic analysis: {e}")
            return self._create_placeholder_visuals('topics')
    
    def create_answer_analysis_visualizations(self):
        """Create visualizations for answer analysis"""
        print("🎯 Creating answer analysis visualizations...")
        
        try:
            # 1. Answer value distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            # Filter reasonable range for better visualization
            reasonable_answers = self.combined_df[
                (self.combined_df['final_answer_numeric'] >= 0) & 
                (self.combined_df['final_answer_numeric'] <= 1000)
            ]['final_answer_numeric']
            
            plt.hist(reasonable_answers, bins=30, alpha=0.7, color='purple', edgecolor='black')
            plt.xlabel('Final Answer Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Final Answers (0-1000 range)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # Integer vs decimal
            integer_count = (self.combined_df['final_answer_numeric'] % 1 == 0).sum()
            decimal_count = len(self.combined_df) - integer_count
            
            plt.pie([integer_count, decimal_count], 
                    labels=['Integer Answers', 'Decimal Answers'], 
                    autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
            plt.title('Integer vs Decimal Answers')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/answer_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'answer_analysis': 'images/gsm8k/answer_analysis.png'            }
            
        except Exception as e:
            print(f"❌ Error in answer analysis: {e}")
            return self._create_placeholder_visuals('answers')
    
    def create_train_validation_comparison(self):
        """Create comprehensive train-validation comparison"""
        print("📈 Creating train-validation comparison...")
        
        try:
            # Comparison metrics
            comparison_data = {
                'Metric': ['Avg Question Length', 'Avg Answer Length', 'Avg Reasoning Steps', 
                          'Addition %', 'Subtraction %', 'Multiplication %', 'Division %'],
                'Train': [
                    self.train_df['question_word_count'].mean(),
                    self.train_df['answer_word_count'].mean(),
                    self.train_df['reasoning_steps'].mean(),
                    self.train_df['has_addition'].mean() * 100,
                    self.train_df['has_subtraction'].mean() * 100,
                    self.train_df['has_multiplication'].mean() * 100,
                    self.train_df['has_division'].mean() * 100
                ],
                'Validation': [
                    self.validation_df['question_word_count'].mean(),
                    self.validation_df['answer_word_count'].mean(),
                    self.validation_df['reasoning_steps'].mean(),
                    self.validation_df['has_addition'].mean() * 100,
                    self.validation_df['has_subtraction'].mean() * 100,
                    self.validation_df['has_multiplication'].mean() * 100,
                    self.validation_df['has_division'].mean() * 100
                ]
            }
            
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(comparison_data['Metric']))
            width = 0.35
            
            plt.bar(x - width/2, comparison_data['Train'], width, label='Train', alpha=0.7)
            plt.bar(x + width/2, comparison_data['Validation'], width, label='Validation', alpha=0.7)
            
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.title('Train vs Validation Set Comparison')
            plt.xticks(x, comparison_data['Metric'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/train_val_detailed_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'train_val_detailed_comparison': 'images/gsm8k/train_val_detailed_comparison.png',
                'comparison_data': comparison_data
            }
            
        except Exception as e:
            print(f"❌ Error in train-validation comparison: {e}")
            return {
                'train_val_detailed_comparison': 'gsm8k/train_val_detailed_comparison.png',
                'comparison_data': {'Metric': [], 'Train': [], 'Validation': []}
            }
    
    def _create_placeholder_visuals(self, viz_type):
        """Create placeholder visualizations when errors occur"""
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'{viz_type.title()} Analysis\n(Visualization not available)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        
        filename = f'{self.output_dir}/{viz_type}_placeholder.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {f'{viz_type}_placeholder': f'images/gsm8k/{viz_type}_placeholder.png'}
    
    def run_complete_analysis(self):
        """Run complete GSM8K dataset analysis"""
        print("🚀 Starting GSM8K Dataset Analysis...")
        
        try:
            # Load dataset
            if not self.load_dataset():
                return None
            
            # Extract features
            if not self.extract_features():
                return None
            
            # Generate summary
            summary = self.generate_dataset_summary()
            
            # Create all visualizations
            text_visuals = self.create_text_length_visualizations()
            reasoning_visuals = self.create_reasoning_analysis_visualizations()
            operation_visuals = self.create_operation_analysis_visualizations()
            topic_visuals = self.create_topic_analysis_visualizations()
            answer_visuals = self.create_answer_analysis_visualizations()
            comparison_results = self.create_train_validation_comparison()
            
            # Combine all results
            self.results = {
                'summary': summary,
                'visuals': {
                    **text_visuals,
                    **reasoning_visuals,
                    **operation_visuals,
                    **topic_visuals,
                    **answer_visuals,
                    'train_val_detailed_comparison': comparison_results['train_val_detailed_comparison']
                },
                'comparison_data': comparison_results['comparison_data'],
                'sample_questions': self.combined_df[['question', 'answer', 'split']].head(10).to_dict('records')
            }
            
            print("✅ GSM8K Analysis Completed Successfully!")
            print(f"📁 Generated {len(self.results['visuals'])} visualizations")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Complete GSM8K analysis failed: {e}")
            return None

# Example usage
if __name__ == "__main__":
    visualizer = GSM8KVisualizer()
    results = visualizer.run_complete_analysis()
    
    if results:
        print(f"\n📊 Dataset Summary:")
        for key, value in results['summary'].items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ Analysis failed!")