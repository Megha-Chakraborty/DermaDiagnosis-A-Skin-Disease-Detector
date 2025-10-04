from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pickle
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import shutil  # Add this import for file copying

app = Flask(__name__)
app.secret_key = "skinDisorderPrediction"

# Load the best model
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'Models')
model_files = [f for f in os.listdir(models_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
if model_files:
    model_path = os.path.join(models_dir, model_files[0])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded: {model_path}")
    
    # Load model info
    try:
        model_summary = pd.read_csv(os.path.join(script_dir, 'Results/model_summary.csv'))
        optimal_threshold = model_summary['threshold'].iloc[0]
        model_metrics = {
            'accuracy': model_summary['accuracy'].iloc[0],
            'precision': model_summary['precision'].iloc[0], 
            'recall': model_summary['recall'].iloc[0],
            'f1_score': model_summary['f1_score'].iloc[0],
            'auc_roc': model_summary['auc_roc'].iloc[0]
        }
    except Exception as e:
        print(f"Error loading model summary: {e}")
        optimal_threshold = 0.5
        model_metrics = {}
else:
    model = None
    optimal_threshold = 0.5
    model_metrics = {}
    print("No model found!")

# Load feature names
try:
    # Try to get feature names from X_test
    X_test = pd.read_csv(os.path.join(script_dir, 'Dataset/processed/X_test.csv'))
    feature_columns = X_test.columns.tolist()
except Exception as e:
    print(f"Could not load feature columns: {e}")
    feature_columns = []

# Function to make prediction
# Replace your predict_melanoma function with this complete version:
def predict_melanoma(input_data, threshold=optimal_threshold):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Extract features for clinical assessment
        features = input_data.iloc[0].to_dict()
        
        # Calculate clinical risk based on ABCDE criteria
        clinical_assessment = calculate_clinical_risk(features)
        
        # Use clinical assessment for all values
        return {
            "model_probability": clinical_assessment["probability"],  # Use clinical value for model too
            "model_prediction": 1 if clinical_assessment["probability"] >= 0.5 else 0,
            "model_risk": clinical_assessment["risk"],
            "clinical_probability": clinical_assessment["probability"],
            "clinical_risk": clinical_assessment["risk"],
            "clinical_risk_score": clinical_assessment["risk_score"],
            "clinical_max_score": clinical_assessment["max_score"],
            "probability": clinical_assessment["probability"],
            "prediction": 1 if clinical_assessment["probability"] >= 0.5 else 0,
            "risk": clinical_assessment["risk"]
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}
# Add this function after your imports
def calculate_clinical_risk(features):
    """
    Calculate clinical risk score based on ABCDE criteria and other symptoms.
    Returns a probability between 0 and 1.
    """
    # Initialize risk score
    risk_score = 0
    max_score = 6  # Maximum possible score
    
    # ABCDE criteria (major factors)
    # A - Asymmetry
    risk_score += 1 if features.get('asymmetry', 0) == 1 else 0
    
    # B - Border irregularity (scale 0-3)
    border_score = int(features.get('border_irregularity', 0))
    # Only count significant border irregularity
    risk_score += 0 if border_score <= 1 else (border_score - 1)
    
    # C - Color variation (scale 0-3)
    color_score = int(features.get('color_variation', 0))
    # Only count multiple colors
    risk_score += 0 if color_score <= 1 else (color_score - 1)
    
    # D - Diameter > 6mm
    risk_score += 1 if features.get('diameter_mm', 0) >= 6 else 0
    
    # E - Evolution
    risk_score += 1.5 if features.get('evolution', 0) == 1 else 0  # Weighted more heavily
    
    # Additional warning signs (minor factors)
    # Only count these if at least one major factor is present
    has_major_factor = (features.get('asymmetry', 0) == 1 or 
                        int(features.get('border_irregularity', 0)) >= 2 or
                        int(features.get('color_variation', 0)) >= 2 or
                        features.get('diameter_mm', 0) >= 6 or
                        features.get('evolution', 0) == 1)
    
    if has_major_factor:
        risk_score += 0.3 if features.get('itchiness', 0) == 1 else 0
        risk_score += 0.5 if features.get('bleeding', 0) == 1 else 0
        risk_score += 0.3 if features.get('pain', 0) == 1 else 0
    
    # Calculate probability (capped at 1.0)
    probability = min(risk_score / max_score, 1.0)
    
    # Determine risk level
    if probability >= 0.6:  # Increased threshold
        risk = "High Risk"
    elif probability >= 0.25:  # Decreased threshold
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    
    return {
        "probability": probability,
        "risk": risk,
        "risk_score": risk_score,
        "max_score": max_score
    }

@app.route('/', methods=['GET'])
def index():
    # Get example features and results
    features = None
    results = None
    risk_descriptions = {
        "High Risk": "Immediate medical consultation recommended",
        "Medium Risk": "Follow-up with dermatologist advised",
        "Low Risk": "Monitor and practice regular skin checks"
    }
    
    # Load summary metrics and figures for the dashboard
    try:
        feature_importance = pd.read_csv(os.path.join(script_dir, 'Results/feature_importance_xgboost_(smote).csv'))
        top_features = feature_importance.head(5).to_dict('records')
    except Exception as e:
        print(f"Could not load feature importance: {e}")
        top_features = []
    
    return render_template('index.html', 
                         model_metrics=model_metrics,
                         top_features=top_features,
                         features=features, 
                         results=results,
                         risk_descriptions=risk_descriptions)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            features = {}
            
            # Basic information
            features['age'] = float(request.form.get('age', 0))
            features['sex'] = int(request.form.get('sex', 0))
            features['localization'] = int(request.form.get('localization', 0))
            features['diameter_mm'] = float(request.form.get('diameter_mm', 0))
            
            # ABCDE criteria
            features['asymmetry'] = int(request.form.get('asymmetry', 0))
            features['border_irregularity'] = int(request.form.get('border_irregularity', 0))
            features['color_variation'] = int(request.form.get('color_variation', 0))
            features['evolution'] = int(request.form.get('evolution', 0))
            
            # Additional symptoms
            features['itchiness'] = int(request.form.get('itchiness', 0))
            features['bleeding'] = int(request.form.get('bleeding', 0))
            features['pain'] = int(request.form.get('pain', 0))
            
            # Add any remaining features your model expects
            # For any missing features, use defaults (like 0)
            for col in feature_columns:
                if col not in features:
                    features[col] = 0
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([features])
            
            # Make prediction
            results = predict_melanoma(input_df)
            
            if "error" in results:
                flash(f"Error: {results['error']}", 'danger')
                return redirect(url_for('predict'))
                
            return render_template('prediction_result.html', results=results, features=features)
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
            return redirect(url_for('predict'))
    
    # GET request - show prediction form
    return render_template('prediction_form.html')

@app.route('/dashboard')
def dashboard():
    # Load summary results
    try:
        model_summary = pd.read_csv(os.path.join(script_dir, 'Results/model_summary.csv'))
        model_name = model_summary['model'].iloc[0]
    except Exception as e:
        print(f"Error loading model summary: {e}")
        model_name = "Unknown Model"
    
    # List all charts in Results directory
    results_dir = os.path.join(script_dir, 'Results')
    static_dir = os.path.join(script_dir, 'static')
    chart_files = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                # Create paths for source and destination
                source_path = os.path.join(results_dir, file)
                dest_path = os.path.join(static_dir, 'charts', file)
                
                # Create charts directory in static if it doesn't exist
                os.makedirs(os.path.join(static_dir, 'charts'), exist_ok=True)
                
                # Copy the file to static/charts
                shutil.copy(source_path, dest_path)
                
                # Add the path relative to static
                chart_files.append(f'charts/{file}')
        
        print(f"Found and copied {len(chart_files)} chart files to static directory")
    else:
        print(f"Results directory not found at {results_dir}")
    
    return render_template('dashboard.html', 
                          model_name=model_name, 
                          model_metrics=model_metrics,
                          chart_files=chart_files)

if __name__ == '__main__':
    app.run(debug=True)
