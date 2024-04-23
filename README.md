# Vehicle Price Prediction: Machine Learning Models for Car Valuation

This project aims to build a machine learning model to predict the price of vehicles based on various features. The dataset used for this project is sourced from Kaggle, containing information about different cars including their selling price, age, fuel type, seller type, and transmission type.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Overview

In this project, three different machine learning algorithms are implemented for vehicle price prediction:

1. Linear Regression
2. Lasso Regression
3. Random Forest Regressor

Each model is trained using the provided dataset and evaluated for its accuracy. The best-performing model can then be used to predict vehicle prices based on input features.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the dependencies using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```


## Usage
1. Clone the repository to your local machine:
```bash
git clone https://github.com/sky9262/vehicle-price-prediction.git
```
2. Navigate to the project directory:
```bash
cd vehicle-price-prediction

```
3. Open the Jupyter Notebook `Vehicle_Price_Prediction.ipynb`:
```bash
jupyter notebook Vehicle_Price_Prediction.ipynb
```
4. Follow the instructions in the notebook to execute each cell and train the models.

5. Once the models are trained, you can use them to make predictions on new data.

## Files Included
- `Vehicle_Price_Prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.

- `car data.csv`: Dataset file containing information about vehicle features and selling prices.
- `README.md`: This file providing an overview of the project and instructions for usage.

- Model files (*.pickle): Saved models for each regression algorithm.


## Results
Each model's performance is evaluated using the coefficient of determination (R² score) on a test dataset. The model with the highest R² score indicates the best performance. Additionally, visualizations are provided to compare the true selling prices with the predicted prices for each model.

Below are the Images of result after training with 3 diffreent algorithms:


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

