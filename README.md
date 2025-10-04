# Early Skin Disease Detection System

## Overview
The Early Skin Disease Detection System is a web-based tool that provides clinical assessment for early skin disorder detection. The system analyzes skin lesion characteristics using both clinical criteria and machine learning to provide risk assessments and recommendations.

## Features
- **Clinical Assessment**: Analyzes skin lesions using ABCDE criteria (Asymmetry, Border irregularity, Color variation, Diameter, Evolution)
- **Risk Evaluation**: Categorizes skin lesions into Low, Medium, or High risk levels
- **Interactive Dashboard**: Displays model performance metrics and feature importance
- **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites
- Python 3.8+
- Flask
- scikit-learn
- pandas
- numpy
- matplotlib
- xgboost

### Setup
1. Clone the repository
```bash
git clone https://github.com/Megha-Chakraborty/Skin-Disease-Detector.git
cd Skin-Disease-Detector
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

4. Open in your browser
```
http://localhost:5000
```

## Usage

### Skin Lesion Assessment
1. Navigate to the "Predict" page
2. Fill out the symptom questionnaire regarding your skin lesion
3. Submit the form to receive a risk assessment

### Viewing Model Performance
1. Navigate to the "Dashboard" page to view model metrics
2. Explore charts showing model performance and feature importance

## Technical Details

### Model
- Trained on the HAM10000 dataset
- Uses clinical ABCDE criteria for assessment
- Key performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC

### Risk Assessment
The system calculates risk based on:
- Asymmetry (1 point)
- Border irregularity (1 point)
- Color variation (1 point)
- Diameter > 6mm (1 point)
- Evolution/change over time (1 point)
- Additional symptoms like bleeding, itching, pain (1 point)

### File Structure
```
SkinDisorder-Prediction/
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
├── Models/                 # Trained ML models
├── Dataset/                # Training and test data
│   └── processed/          # Processed dataset
├── Results/                # Analysis results and charts
├── static/                 # Static files (CSS, JS, images)
└── templates/              # HTML templates
    ├── index.html          # Home page
    ├── predict.html        # Prediction form
    ├── prediction_result.html  # Results page
    └── dashboard.html      # Performance dashboard
```

## Limitations
- This tool is for educational purposes only
- Not a substitute for professional medical diagnosis
- Always consult with a dermatologist for proper diagnosis

## How It Works

### Clinical Assessment
The system uses the widely accepted ABCDE criteria for assessing skin lesions:
- **A**symmetry: One half unlike the other half
- **B**order: Irregular, scalloped or poorly defined border
- **C**olor: Varies from one area to another
- **D**iameter: Usually greater than 6mm
- **E**volution: Change in size, shape, color, or symptoms

### Risk Calculation
Risk is calculated based on the presence and severity of each criterion. The system then categorizes lesions into:
- **Low Risk**: Score of 0-1.5 points
- **Medium Risk**: Score of 2-3.5 points
- **High Risk**: Score of 4-6 points

## Future Improvements
- Integration with mobile camera for direct image analysis
- Expanded dataset for improved accuracy
- Multi-class classification for different skin disease types
- Patient history tracking and follow-up reminders

## Acknowledgments
- HAM10000 dataset for providing training data
- Flask framework for web application development
- Bootstrap for responsive UI design

## Contact
For questions or support, please contact [ch.megha0401@gmail.com](mailto:ch.megha0401@gmail.com)
