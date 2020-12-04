#!/usr/bin/env python
# coding: utf-8

# # Used Car Prices
# 
# ### Problem Statement
# The aim of this project is to create regression model to help the new car trader company determine the price of used cars.
# ### Evaluation Metric
# Mean squared error (𝑀𝑆𝐸)

# Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.base import TransformerMixin, BaseEstimator

# Create a class to transform columns 
class FeatureEngineering(BaseEstimator, TransformerMixin):
       
    def fit(self, X, y=None):
        return self  
    
    def transform(self, X, y=None):
        df = X
        
        # Create 'name' variable to combine manufacture and model names
        columns_strip = ['manufacturer_name', 'model_name']
        # Delete extra space in strings
        for column in columns_strip:
            df[column] = df[column].apply(lambda x: x.strip())
        # Combine manufacture and model names    
        df['name'] = df['manufacturer_name'] + ' ' + df['model_name']

        # Create a feature that represents mileage per year
        df['odometer_value/year'] = round(df['odometer_value']/(2020 - df['year_produced']))
        # Create a feature how old is a car
        df['year'] = 2020 - df['year_produced']

        # Reduce the number of car model names
        # Set a limit of rare car occurrence
        car_total = 6
        # Count a number of car names and convert the result to a dataframe
        car_models = pd.DataFrame(df['name'].value_counts())
        # Get a list of rare car names
        car_models_list = car_models[car_models['name'] < car_total].index
        # create a new category'other' for rare car model names
        df['name'] = df['name'].apply(lambda x: 'other' if x in car_models_list else x)

        # Create features to reduce a number of categories
        hybrid ='hybrid_or_electric'
        df['engine_fuel'] = df['engine_fuel'].replace({'hybrid-petrol':hybrid,'hybrid-diesel':hybrid,'electric':hybrid})

        # Create a list of unnamed features
        features_list = ['feature_0','feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
        # Count a total number of unnamed features for a car
        df['other_features']=df[features_list].sum(axis=1)
        
        global feats
        feats = ['name', 'odometer_value/year', 'year', 'other_features'] 
        
        return X
    
# Load the data
prediction_data = '2. Prepared Data/pred_cars.csv'
df = pd.read_csv(prediction_data)

# Load a feature engineering pipeline from a file
features_path = '5. Insights/Models/used_car_prices_feature_engineering.pickle'
with open(features_path, 'rb') as f:
    feature_engineering_loaded = pickle.load(f)
print('Regression model {} is loaded.'.format(feature_engineering_loaded))

# Load a model from a file
model_path = '5. Insights/Models/used_car_prices_model.pickle'
with open(model_path, 'rb') as f:
    regression_model_loaded = pickle.load(f)
print('Regression model {} is loaded.'.format(regression_model_loaded))

# Load the PorewTransformer object for the inverse target transformation
transformer_path = '5. Insights/Models/used_car_prices_target_transformation.pickle'
with open(transformer_path, 'rb') as f:
    transformer_loaded = pickle.load(f)
print('Transformer {} is loaded.'.format(transformer_loaded))

# Define predictor variables
features =[ 'manufacturer_name', 'has_warranty', 'state', 'drivetrain', 'transmission', 'name',
           'odometer_value', 'odometer_value/year', 'year',  'engine_fuel', 'color',
           'duration_listed', 'body_type', 'engine_capacity', 'other_features', 'feature_0',
           ]


# ### ---------------------------------------- Prediction ------------------------------------------
# Create new features
df=feature_engineering_loaded.fit_transform(df)

# Make prediction using the pipeline
prediction = regression_model_loaded.predict(df[features])

# Transform predicted price to get the rounded car price in dollars
y_predict_price = transformer_loaded.inverse_transform(prediction.reshape(-1,1))
# Round car price to hundred
y_predict_price_round = np.round(y_predict_price,-2)
# Create a dataframe with results
results = pd.DataFrame( {'Predicted':y_predict_price.reshape(-1),
                         'Predicted rounded':y_predict_price_round.reshape(-1)},
                          index=df.index)

# Form a dataframe with car information and predicted prices
prediction_results = df.join(results)

# Save car information and predicted prices to csv file
fl = "5. Insights/Prediction/used_car_prices_prediction_data_predicted_price.csv"
prediction_results.to_csv(fl, index=False)

# Save predicted prices to csv file
fl = "5. Insights/Prediction/used_car_prices_predicted_price.csv"
results.to_csv(fl, index=False)