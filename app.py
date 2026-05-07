import sys
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


def validate_input(data):
    """Validate user input data.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    validations = CONFIG.validations
    
    # Validate gender
    if data.get('gender') not in validations.valid_genders:
        return False, f"Invalid gender. Must be one of: {validations.valid_genders}"
    
    # Validate race/ethnicity
    if data.get('ethnicity') not in validations.valid_race_ethnicity:
        return False, f"Invalid ethnicity. Must be one of: {validations.valid_race_ethnicity}"
    
    # Validate parental education
    if data.get('parental_level_of_education') not in validations.valid_parental_education:
        return False, f"Invalid parental education. Must be one of: {validations.valid_parental_education}"
    
    # Validate lunch
    if data.get('lunch') not in validations.valid_lunch:
        return False, f"Invalid lunch type. Must be one of: {validations.valid_lunch}"
    
    # Validate test preparation
    if data.get('test_preparation_course') not in validations.valid_test_preparation:
        return False, f"Invalid test preparation. Must be one of: {validations.valid_test_preparation}"
    
    # Validate reading score
    try:
        reading_score = float(data.get('reading_score'))
        if not (validations.score_min <= reading_score <= validations.score_max):
            return False, f"Reading score must be between {validations.score_min} and {validations.score_max}"
    except (ValueError, TypeError):
        return False, "Reading score must be a number"
    
    # Validate writing score
    try:
        writing_score = float(data.get('writing_score'))
        if not (validations.score_min <= writing_score <= validations.score_max):
            return False, f"Writing score must be between {validations.score_min} and {validations.score_max}"
    except (ValueError, TypeError):
        return False, "Writing score must be a number"
    
    return True, None


# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get form data
            form_data = {
                'gender': request.form.get('gender'),
                'ethnicity': request.form.get('ethnicity'),
                'parental_level_of_education': request.form.get('parental_level_of_education'),
                'lunch': request.form.get('lunch'),
                'test_preparation_course': request.form.get('test_preparation_course'),
                'reading_score': request.form.get('reading_score'),
                'writing_score': request.form.get('writing_score')
            }
            
            # Validate input
            is_valid, error_message = validate_input(form_data)
            if not is_valid:
                logging.warning(f"Input validation failed: {error_message}")
                return render_template('home.html', error=error_message)
            
            data = CustomData(
                gender=form_data['gender'],
                race_ethnicity=form_data['ethnicity'],
                parental_level_of_education=form_data['parental_level_of_education'],
                lunch=form_data['lunch'],
                test_preparation_course=form_data['test_preparation_course'],
                reading_score=float(form_data['reading_score']),
                writing_score=float(form_data['writing_score'])
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")

            results = predict_pipeline.predict(pred_df)
            print("after Prediction")

            return render_template('home.html', results=results[0])
        
        except CustomException as e:
            logging.error(f"Pipeline error during prediction: {e.error_message}")
            return render_template(
                'home.html',
                error="Prediction failed due to data processing error. Please check your inputs and try again."
            )
        except Exception as e:
            logging.error(f"Unexpected error during prediction: {e}", exc_info=True)
            return render_template(
                'home.html',
                error="An unexpected error occurred. Please try again later."
            )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON input).
    
    Request body example:
    {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": 72,
        "writing_score": 74
    }
    
    Response example:
    {
        "prediction": 72.5,
        "status": "success"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400
        
        # Validate input
        form_data = {
            'gender': data.get('gender'),
            'ethnicity': data.get('race_ethnicity'),
            'parental_level_of_education': data.get('parental_level_of_education'),
            'lunch': data.get('lunch'),
            'test_preparation_course': data.get('test_preparation_course'),
            'reading_score': data.get('reading_score'),
            'writing_score': data.get('writing_score')
        }
        
        is_valid, error_message = validate_input(form_data)
        if not is_valid:
            return jsonify({"error": error_message, "status": "error"}), 400
        
        custom_data = CustomData(
            gender=form_data['gender'],
            race_ethnicity=form_data['ethnicity'],
            parental_level_of_education=form_data['parental_level_of_education'],
            lunch=form_data['lunch'],
            test_preparation_course=form_data['test_preparation_course'],
            reading_score=float(form_data['reading_score']),
            writing_score=float(form_data['writing_score'])
        )
        
        pred_df = custom_data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return jsonify({
            "prediction": float(results[0]),
            "status": "success"
        })
    
    except CustomException as e:
        logging.error(f"API pipeline error: {e.error_message}")
        return jsonify({
            "error": "Prediction failed due to invalid input or processing error.",
            "status": "error"
        }), 400
    except Exception as e:
        logging.error(f"Unexpected API error: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error. Please try again later.",
            "status": "error"
        }), 500


if __name__=="__main__":
    app.run(host="0.0.0.0")
