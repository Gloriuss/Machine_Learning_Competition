# Predicting Water Pump Status

## Overview
This project showcases my participation in the “Can you predict which water pumps are faulty?” machine learning competition. The goal of the competition was to predict the status of water pumps in Tanzania using data provided by Taarifa and the Tanzanian Ministry of Water. The target variable consists of three categories:
	1.	Functional
	2.	Needs Repair
	3.	Non-functional

A predictive model can help prioritize maintenance and ensure communities have access to clean water.

## Project Structure
.
├── Machine_Learning_Competition.py  # Main Python script
├── Notebook.ipynb                  # Jupyter Notebook for analysis
├── requirements.txt                # Python dependencies
├── Data_Download                   # Raw datasets
│   ├── SubmissionFormat.csv
│   ├── test_set_values.csv
│   ├── training_set_labels.csv
│   └── training_set_values.csv
└── Prediction_Result               # Generated predictions
    └── y_model_predictions.csv

## How to Reproduce

### Steps to Run
	1.	Clone this repository:
      git clone [repository_url](https://github.com/Gloriuss/Machine_Learning_Competition.git)
      cd Machine_Learning_Competition
 	2.	raw datasets in the Data_Download folder are allready inside GIT.
  3.  Install all required packages using:
      pip install -r requirements.txt
	4.	Run the main script:
      python Machine_Learning_Competition.py
 	5.	The predictions will be saved in Prediction_Result/y_model_predictions.csv.

## Approach

### Data Handling

	•	Combining Datasets: Training and testing datasets were concatenated to ensure consistent preprocessing.
	•	Imputation: Missing values were handled using the KNN Imputer.
	•	Categorical Encoding: Categorical variables were numerically encoded to facilitate model training.
	•	Feature Engineering:
	•	Dates were transformed into days since the last recorded date.
	•	Boolean features were converted into numerical representations.
	•	Columns with only one unique value were removed to reduce noise.

### Model Selection

	•	The Random Forest Classifier was chosen for its robustness in handling categorical and numerical data.

### Training and Prediction

	1.	The model was trained on the preprocessed training dataset.
	2.	Predictions were generated for the test dataset and mapped back to the original class labels.
	3.	The results were saved in the submission format provided.

### Results

	•	The predictions file (y_model_predictions.csv) is formatted as required by the competition and contains:
	•	id: Water pump identifiers from the test set.
	•	status_group: Predicted class label (functional, needs repair, non-functional).

### Key Learnings

### This competition provided hands-on experience with:
	•	Real-world data cleaning and preprocessing.
	•	Handling imbalanced datasets.
	•	Designing a machine learning pipeline from data preparation to result generation.
