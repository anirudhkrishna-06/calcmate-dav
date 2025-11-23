from flask import Flask, render_template, request, jsonify
import sys
import os

# Add visualizations directory to path
sys.path.append('visualizations')

from visualizations.pipeline_viz import PipelineVisualizer
from visualizations.user_interactions_viz import UserInteractionsVisualizer
from visualizations.problem_bank_viz import ProblemBankVisualizer

app = Flask(__name__)

# Global variables to store analysis results
pipeline_results = {}
interactions_results = {}
problem_bank_results = {}

def run_pipeline_analysis():
    """Run the pipeline analysis and cache results"""
    global pipeline_results
    try:
        visualizer = PipelineVisualizer('static/datasets/pipeline_diagnostics.csv')
        results = visualizer.run_complete_analysis()
        if results:
            pipeline_results = results
            return True
        return False
    except Exception as e:
        print(f"Error in pipeline analysis: {e}")
        return False

def run_interactions_analysis():
    """Run the user interactions analysis and cache results"""
    global interactions_results
    try:
        visualizer = UserInteractionsVisualizer('static/datasets/user_interactions.csv')
        results = visualizer.run_complete_analysis()
        if results:
            interactions_results = results
            return True
        return False
    except Exception as e:
        print(f"Error in interactions analysis: {e}")
        return False

def run_problem_bank_analysis():
    """Run the problem bank analysis and cache results"""
    global problem_bank_results
    try:
        visualizer = ProblemBankVisualizer('static/datasets/problem_bank.csv')
        results = visualizer.run_complete_analysis()
        if results:
            problem_bank_results = results
            return True
        return False
    except Exception as e:
        print(f"Error in problem bank analysis: {e}")
        return False

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

# ==================== PIPELINE DIAGNOSTICS ROUTES ====================

@app.route('/pipeline')
def pipeline_dashboard():
    """Main pipeline diagnostics dashboard"""
    if not pipeline_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running pipeline analysis", 500
    
    return render_template(
        'pipeline.html',
        overview=pipeline_results['overview'],
        eda=pipeline_results['eda'],
        visuals=pipeline_results['visuals'],
        insights=pipeline_results['insights']
    )

@app.route('/pipeline/regression')
def pipeline_regression():
    """Detailed regression analysis page for pipeline"""
    if not pipeline_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running pipeline analysis", 500
    
    return render_template(
        'pipeline_regression.html',
        overview=pipeline_results['overview'],
        regression=pipeline_results['regression'],
        regression_plot=pipeline_results['visuals']['regression_plot']
    )

@app.route('/pipeline/hypothesis')
def pipeline_hypothesis():
    """Detailed hypothesis testing page for pipeline"""
    if not pipeline_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running pipeline analysis", 500
    
    return render_template(
        'pipeline_hypothesis.html',
        overview=pipeline_results['overview'],
        hypothesis=pipeline_results['hypothesis'],
        hypothesis_plot=pipeline_results['visuals']['hypothesis_plot']
    )

# ==================== USER INTERACTIONS ROUTES ====================

@app.route('/interactions')
def interactions_dashboard():
    """Main user interactions dashboard"""
    if not interactions_results:
        success = run_interactions_analysis()
        if not success:
            return "Error running interactions analysis", 500
    
    return render_template(
        'interactions.html',
        overview=interactions_results['overview'],
        eda=interactions_results['eda'],
        visuals=interactions_results['visuals'],
        insights=interactions_results['insights']
    )

@app.route('/interactions/regression')
def interactions_regression():
    """Detailed regression analysis page for user interactions"""
    if not interactions_results:
        success = run_interactions_analysis()
        if not success:
            return "Error running interactions analysis", 500
    
    return render_template(
        'interactions_regression.html',
        regression=interactions_results['regression'],
        regression_plot=interactions_results['visuals']['regression_plot']
    )

@app.route('/interactions/hypothesis')
def interactions_hypothesis():
    """Detailed hypothesis testing page for user interactions"""
    if not interactions_results:
        success = run_interactions_analysis()
        if not success:
            return "Error running interactions analysis", 500
    
    return render_template(
        'interactions_hypothesis.html',
        overview=interactions_results['overview'],
        hypothesis=interactions_results['hypothesis'],
        hypothesis_plot=interactions_results['visuals']['hypothesis_plot']
    )

# ==================== PROBLEM BANK ROUTES ====================

@app.route('/problem_bank')
def problem_bank_dashboard():
    """Main problem bank dashboard"""
    if not problem_bank_results:
        success = run_problem_bank_analysis()
        if not success:
            return "Error running problem bank analysis", 500
    
    return render_template(
        'problem_bank.html',
        overview=problem_bank_results['overview'],
        eda=problem_bank_results['eda'],
        visuals=problem_bank_results['visuals'],
        insights=problem_bank_results['insights']
    )

@app.route('/problem_bank/regression')
def problem_bank_regression():
    """Detailed regression analysis page for problem bank"""
    if not problem_bank_results:
        success = run_problem_bank_analysis()
        if not success:
            return "Error running problem bank analysis", 500
    
    return render_template(
        'problem_bank_regression.html',
        overview=problem_bank_results['overview'],
        regression=problem_bank_results['regression'],
        actual_predicted=problem_bank_results['visuals']['regression_actual_predicted'],
        coefficients=problem_bank_results['visuals']['regression_coefficients']
    )

@app.route('/problem_bank/hypothesis')
def problem_bank_hypothesis():
    """Detailed hypothesis testing page for problem bank"""
    if not problem_bank_results:
        success = run_problem_bank_analysis()
        if not success:
            return "Error running problem bank analysis", 500
    
    return render_template(
        'problem_bank_hypothesis.html',
        overview=problem_bank_results['overview'],
        hypothesis=problem_bank_results['hypothesis'],
        hypothesis_plot=problem_bank_results['visuals']['hypothesis_plot']
    )

# ==================== DEBUG AND UTILITY ROUTES ====================

@app.route('/debug_data')
def debug_data():
    """Debug route to check data issues"""
    visualizer = UserInteractionsVisualizer('static/datasets/user_interactions.csv')
    if visualizer.load_data():
        visualizer.debug_data_issues()
        return "Check console for data debug info"
    else:
        return "Failed to load data for debugging"

@app.route('/debug_regression')
def debug_regression():
    """Debug route to test regression directly"""
    try:
        visualizer = UserInteractionsVisualizer('static/datasets/user_interactions.csv')
        
        if not visualizer.load_data():
            return "Failed to load data"
        
        # Run regression analysis
        regression_results, regression_plot = visualizer.perform_regression_analysis()
        
        return f"""
        <h1>Regression Debug Results</h1>
        <pre>
        R-squared: {regression_results['r_squared']}
        RMSE: {regression_results['rmse']}
        MAE: {regression_results['mae']}
        Samples: {regression_results['n_samples']}
        Features: {regression_results['n_features']}
        Equation: {regression_results['model_equation']}
        Coefficients: {regression_results['coefficients']}
        </pre>
        """
    except Exception as e:
        return f"Regression debug failed: {str(e)}"

# ==================== API ENDPOINTS ====================

@app.route('/api/refresh-all')
def refresh_all_analyses():
    """API endpoint to refresh all analyses"""
    global pipeline_results, interactions_results, problem_bank_results
    
    pipeline_results = {}
    interactions_results = {}
    problem_bank_results = {}
    
    pipeline_success = run_pipeline_analysis()
    interactions_success = run_interactions_analysis()
    problem_bank_success = run_problem_bank_analysis()
    
    if pipeline_success and interactions_success and problem_bank_success:
        return jsonify({
            "status": "success", 
            "message": "All analyses refreshed successfully"
        })
    else:
        return jsonify({
            "status": "partial",
            "message": f"Pipeline: {'✓' if pipeline_success else '✗'}, "
                      f"Interactions: {'✓' if interactions_success else '✗'}, "
                      f"Problem Bank: {'✓' if problem_bank_success else '✗'}"
        }), 500

@app.route('/api/pipeline/stats')
def get_pipeline_stats():
    """API endpoint to get pipeline statistics"""
    if not pipeline_results:
        return jsonify({"status": "error", "message": "No pipeline data"}), 404
    return jsonify({"status": "success", "data": pipeline_results['overview']})

@app.route('/api/interactions/stats')
def get_interactions_stats():
    """API endpoint to get user interactions statistics"""
    if not interactions_results:
        return jsonify({"status": "error", "message": "No interactions data"}), 404
    return jsonify({"status": "success", "data": interactions_results['overview']})

@app.route('/api/problem_bank/stats')
def get_problem_bank_stats():
    """API endpoint to get problem bank statistics"""
    if not problem_bank_results:
        return jsonify({"status": "error", "message": "No problem bank data"}), 404
    return jsonify({"status": "success", "data": problem_bank_results['overview']})

# Add to imports at the top
from visualizations.gsm8k_viz import GSM8KVisualizer

# Add global variable
gsm8k_results = {}

def run_gsm8k_analysis():
    """Run GSM8K analysis and cache results"""
    global gsm8k_results
    
    try:
        visualizer = GSM8KVisualizer()
        results = visualizer.run_complete_analysis()
        
        if results:
            gsm8k_results = results
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in GSM8K analysis: {e}")
        return False

# Add new routes
@app.route('/gsm8k')
def gsm8k_dashboard():
    """GSM8K dataset analytics dashboard"""
    if not gsm8k_results:
        success = run_gsm8k_analysis()
        if not success:
            return "Error running GSM8K analysis", 500
    
    return render_template(
        'gsm8k.html',
        summary=gsm8k_results['summary'],
        visuals=gsm8k_results['visuals'],
        comparison_data=gsm8k_results['comparison_data'],
        sample_questions=gsm8k_results['sample_questions']
    )

@app.route('/api/gsm8k/refresh')
def refresh_gsm8k():
    """API endpoint to refresh GSM8K analysis"""
    global gsm8k_results
    gsm8k_results = {}
    
    success = run_gsm8k_analysis()
    if success:
        return jsonify({"status": "success", "message": "GSM8K analysis refreshed"})
    else:
        return jsonify({"status": "error", "message": "GSM8K analysis failed"}), 500

if __name__ == '__main__':
    # Run initial analyses when app starts
    print("🚀 Starting Flask application...")
    print("=" * 70)
    
    print("\n📊 Dataset 1: Running Pipeline Diagnostics analysis...")
    if run_pipeline_analysis():
        print("✅ Pipeline analysis completed successfully!")
    else:
        print("❌ Pipeline analysis failed!")
    
    print("\n👥 Dataset 2: Running User Interactions analysis...")
    if run_interactions_analysis():
        print("✅ Interactions analysis completed successfully!")
    else:
        print("❌ Interactions analysis failed!")
    
    print("\n📚 Dataset 3: Running Problem Bank analysis...")
    if run_problem_bank_analysis():
        print("✅ Problem Bank analysis completed successfully!")
    else:
        print("❌ Problem Bank analysis failed!")
    
    # ADD THIS - Run GSM8K at startup
    print("\n🧮 Dataset 4: Running GSM8K analysis...")
    if run_gsm8k_analysis():
        print("✅ GSM8K analysis completed successfully!")
    else:
        print("❌ GSM8K analysis failed!")
    
    print("\n" + "=" * 70)
    print("🌐 Starting Flask server...")
    print("📍 Navigate to: http://localhost:5000")
    print("=" * 70 + "\n")
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
