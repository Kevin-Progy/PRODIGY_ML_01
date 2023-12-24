# PRODIGY_ML_01
# House Price Prediction

This repository contains a simple linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms. The code is implemented in Python using the scikit-learn library.

## Getting Started

Follow these steps to use the code in this repository:

1. **Clone the Repository:**
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

2. **Install Dependencies:**

3. **Upload Dataset:**
- Download the dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
- Upload the downloaded dataset file (`train.csv`) to the repository.

4. **Upload Kaggle API Key:**
- If you don't have a Kaggle API key, create one from your Kaggle account settings.
- Upload the Kaggle API key (`kaggle.json`) to the repository.

5. **Run the Code:**
- Execute the Jupyter notebook or Python script to train the linear regression model and make predictions.

## File Structure

- `house_price_prediction.ipynb`: Jupyter notebook containing the code.
- `requirements.txt`: List of Python dependencies.

## Model Evaluation

After training the model, the script evaluates its performance using Mean Squared Error (MSE) and R-squared metrics. Additionally, it plots the predictions against actual prices.

## Test a Prediction

At the end of the script, there is code to test a prediction for a new house. Provide the features (square footage, bedrooms, bathrooms) for the new house to get the predicted price.

```python
# Test a prediction for a new house
new_house_features = np.array([[2000, 3, 2]])  # Provide values for GrLivArea, BedroomAbvGr, FullBath
predicted_price = model.predict(new_house_features)

print(f'Predicted Price for the New House: ${predicted_price[0]:,.2f}')
