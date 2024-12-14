# Car Price Prediction

This project focuses on building a machine learning model to predict car prices based on various features. The primary objective is to help a Chinese automobile company entering the US market understand the pricing dynamics of cars in the American market. This model will assist in analyzing significant factors affecting car prices and provide insights for strategic decisions.

## Dataset
The dataset contains information about various cars in the American market. It includes numerical and categorical features such as car specifications and their respective prices.

### Dataset Link
[Download Dataset](https://drive.google.com/uc?id=1FHmYNLs9v0Enc-UExEMpitOFGsWvB2dP)

## Steps in the Project

### 1. Load and Inspect the Dataset
- Load the dataset from the provided link.
- Perform initial data inspection to understand the structure and basic statistics.

### 2. Preprocess the Data
- Handle missing values by removing incomplete rows.
- Encode categorical variables using label encoding.
- Scale numerical features using standard scaling.
- Split the dataset into training and testing sets.

### 3. Implement Regression Models
The following regression models are trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor

### 4. Evaluate Model Performance
Models are evaluated using the following metrics:
- R-squared
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### 5. Feature Importance Analysis
Feature importance is analyzed using the Random Forest model to identify significant factors affecting car prices.

### 6. Hyperparameter Tuning
Hyperparameter tuning is performed for the Random Forest model using GridSearchCV to improve performance.

## Results
The model evaluation results are saved in a CSV file named `model_results.csv`. The best-performing model is highlighted, and its performance is reported after hyperparameter tuning.

## How to Run
1. Clone the repository.
2. Ensure Python and necessary libraries (listed below) are installed.
3. Download the dataset from the link provided and place it in the project directory.
4. Run the Jupyter Notebook to execute the steps.

## Dependencies
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Files in Repository
- `car_price_prediction.ipynb`: The main Jupyter Notebook containing the code.
- `model_results.csv`: Results of the model evaluation.
- `Car_price_assignment.csv`: data set file

