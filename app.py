from flask import Flask, render_template, request, jsonify
import sys
import os

# Add visualizations directory to path
sys.path.append('visualizations')

from visualizations.pipeline_viz import PipelineVisualizer

app = Flask(__name__)

# Global variable to store analysis results (in production, use caching)
analysis_results = {}

def run_pipeline_analysis():
    """Run the pipeline analysis and cache results"""
    global analysis_results
    
    try:
        visualizer = PipelineVisualizer('static/datasets/pipeline_diagnostics.csv')
        results = visualizer.run_complete_analysis()
        
        if results:
            analysis_results = results
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in pipeline analysis: {e}")
        return False

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/pipeline')
def pipeline_dashboard():
    """Main pipeline diagnostics dashboard"""
    # Run analysis if not already done
    if not analysis_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running analysis", 500
    
    return render_template(
        'pipeline.html',
        overview=analysis_results['overview'],
        eda=analysis_results['eda'],
        visuals=analysis_results['visuals'],
        insights=analysis_results['insights']
    )

@app.route('/pipeline/regression')
def pipeline_regression():
    """Detailed regression analysis page"""
    if not analysis_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running analysis", 500
    
    return render_template(
        'pipeline_regression.html',
        overview=analysis_results['overview'],
        regression=analysis_results['regression'],
        regression_plot=analysis_results['visuals']['regression_plot']
    )

@app.route('/pipeline/hypothesis')
def pipeline_hypothesis():
    """Detailed hypothesis testing page"""
    if not analysis_results:
        success = run_pipeline_analysis()
        if not success:
            return "Error running analysis", 500
    
    return render_template(
        'pipeline_hypothesis.html',
        overview=analysis_results['overview'],
        hypothesis=analysis_results['hypothesis'],
        hypothesis_plot=analysis_results['visuals']['hypothesis_plot']
    )

@app.route('/api/pipeline/refresh')
def refresh_analysis():
    """API endpoint to refresh analysis"""
    global analysis_results
    analysis_results = {}  # Clear cache
    
    success = run_pipeline_analysis()
    if success:
        return jsonify({"status": "success", "message": "Analysis refreshed"})
    else:
        return jsonify({"status": "error", "message": "Analysis failed"}), 500

@app.route('/api/pipeline/stats')
def get_pipeline_stats():
    """API endpoint to get pipeline statistics"""
    if not analysis_results:
        return jsonify({"status": "error", "message": "No analysis data"}), 404
    
    return jsonify({
        "status": "success",
        "data": {
            "overview": analysis_results['overview'],
            "regression": analysis_results['regression'],
            "hypothesis": analysis_results['hypothesis']
        }
    })

if __name__ == '__main__':
    # Run initial analysis when app starts
    print("🚀 Starting Flask application...")
    print("📊 Running initial pipeline analysis...")
    run_pipeline_analysis()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)