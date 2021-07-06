#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

# Create a class to perform feature engineering 
class FeatureEngineering(BaseEstimator, TransformerMixin):
       
    def fit(self, X, y=None):
        return self  
    
    def transform(self, X, y=None):
        df = X
        # Convert boolean values to integer
        df['engine_has_gas']=df['engine_has_gas'].astype(int)
        df['has_warranty']=df['has_warranty'].astype(int)

        # Create a feature that represents mileage per year
        df['odometer_value']=df['odometer_value'].apply(lambda x: round(x, -2))
        df['odometer_value/year'] = round(df['odometer_value']/(2020 - df['year_produced']),-2)
        # Create a feature how old is a car
        df['year'] = 2020 - df['year_produced']

        ################################# Reduce a number of categories  ######################################
    
        # Combine manufacturer and model names 
        df['name'] = df['manufacturer_name'].apply(lambda x: x.strip()) + ' ' + df['model_name'].apply(lambda x: x.strip())

        # Reduce the number of car model names
        # Set a limit of rare car occurrence
        car_total = 10
        # Count a number of car names and convert the result to a dataframe
        car_models = pd.DataFrame(df['manufacturer_name'].value_counts())
        # Get a list of rare car names
        car_models_list = car_models[car_models['manufacturer_name'] < car_total].index
        # create a new category'other' for rare car model names
        df['manufacturer_name'] = df['manufacturer_name'].apply(lambda x: 'other' if x in car_models_list else x)
        
        # Reduce the number of car model names
        # Set a limit of rare car occurrence
        car_total = 10
        # Count a number of car names and convert the result to a dataframe
        car_models = pd.DataFrame(df['model_name'].value_counts())
        # Get a list of rare car names
        car_models_list = car_models[car_models['model_name'] < car_total].index
        # create a new category'other' for rare car model names
        df['model_name'] = df['model_name'].apply(lambda x: 'other' if x in car_models_list else x)
        """
        # Reduce the number of car model names
        # Set a limit of rare car occurrence
        car_total = 20
        # Count a number of car names and convert the result to a dataframe
        car_models = pd.DataFrame(df['name'].value_counts())
        # Get a list of rare car names
        car_models_list = car_models[car_models['name'] < car_total].index
        # create a new category'other' for rare car model names
        df['name'] = df['name'].apply(lambda x: 'other' if x in car_models_list else x)
        
        # Reduce the number of colors
        df['color']=df['color'].replace({'violet':'other','yellow':'other','orange':'other', 'brown':'other'})

        # Reduce the number of body tipes 
        df['body_type']=df['body_type'].replace({'pickup':'other','cabriolet':'other','limousine':'other'})

        # Add 'hybrid-diesel' to 'diesel' category
        df['engine_fuel']=df['engine_fuel'].replace({'hybrid-diesel':'diesel'})
        """
        ################################# End Reduce a number of categories  ######################################
        
        
        # Create a list of unnamed features
        features_list = ['feature_0','feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
        
        for feature in features_list:
            df[feature]=df[feature].astype(int)
        # Count a total number of unnamed features for a car
        df['other_features']=df[features_list].sum(axis=1).astype(int)
        
        global feats
        feats = ['name', 'odometer_value/year', 'year', 'other_features'] 
        
        X=df
        return X
