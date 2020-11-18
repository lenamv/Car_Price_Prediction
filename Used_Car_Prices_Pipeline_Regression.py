#!/usr/bin/env python
# coding: utf-8

# # Used Car Prices
# 
# ### Problem Statement
# The aim of this project is to create regression model to help the new car trader company determine the price of used cars.
# 
# ### Evaluation Metric
# Mean squared error (𝑀𝑆𝐸)

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
import pickle

###################################### Load data #################################################
test_data = '2. Prepared Data/public_cars.csv'
prediction_data = '2. Prepared Data/pred_cars.csv'
df = pd.read_csv(test_data)
df_pred = pd.read_csv(prediction_data)


print("The train dataset rows: {} , columns: {} ".format(df.shape[0],df.shape[1]))
print(df.shape)

print("The test dataset rows: {} , columns: {} ".format(df_pred.shape[0],df_pred.shape[1]))
print(df_pred.shape)
print(df.info())

###################################### Preprocessing data ######################################

# Check duplicated rows
duplicated_rows = df.duplicated().sum()
print('Dataset: {}'.format(df.shape))

# Drop duplicated rows
if duplicated_rows != 0:
    df.drop_duplicates(inplace=True)
print('Dropped duplicated rows : {}'.format(duplicated_rows))
print('Dataset: {}'.format(df.shape))

# Find missing values 
print(f'Missing values: {df.isnull().sum().sum()}')

isnull_filter = df['price_usd'].isnull()

# Drop the row with missing price
missing_price = len(df[isnull_filter].index)
print('\nDropped rows with missing price: {} '.format(missing_price))
df.drop(df[isnull_filter].index, inplace=True)
df = df.reset_index(drop=True)
print('Dataset: {}'.format(df.shape))

# Find missing values 
print('\nMissing values: {}'.format(df.isnull().sum().sum()))

# ### Dealing with outliers
# Drop electric cars
electric_cars = df['engine_type'] == 'electric'
electric = df[electric_cars]

# Remove prices over the 99.5% percentile
quantile = df['price_usd'].quantile(0.995)
price_quantile = df['price_usd'] > quantile
price = df[price_quantile]

outliers = df[(electric_cars | price_quantile )]

# Save the rows with outliers to csv file
fl = "4. Analysis/used_car_prices_outliers.csv"
outliers.to_csv(fl, index=False)

print('Dataset: {}'.format(df.shape))
df.drop( outliers.index, inplace=True)
df.reset_index(drop=True, inplace=True)
print('Dropped rows with electric cars: {} '.format(len(electric)))
print('\n0.995 quantile: {}'.format(quantile))
print('Drop prices over the 99.5% percentile: {}'.format(len(price)))
print('Dataset: {}'.format(df.shape))

# Drop extreme outliers

min_price = 100
mask_price = df['price_usd']<100

max_odometer_value = 500000
max_price = 30000
mask_odometer_value =  (df['odometer_value'] > max_odometer_value) & (df['price_usd'] > max_price)

min_year_produced = 1970
max_price = 14000
mask_produced_price = (df['year_produced'] < min_year_produced) & (df['price_usd'] > max_price)

min_year_produced = df['year_produced'].min()
mask_year_produced_min = (df['year_produced'] == min_year_produced)

min_engine_capacity = 0.8
max_engine_capacity = df['engine_capacity'].max()
mask_engine = (df['engine_capacity']<min_engine_capacity) | (df['engine_capacity'] == max_engine_capacity)

max_duration_listed = df['duration_listed'].max()
mask_duration_listed = (df['duration_listed'] == max_duration_listed)

outliers = df[(mask_price | mask_odometer_value | mask_produced_price | mask_year_produced_min | mask_engine | mask_duration_listed)]

# Save the rows with extreme outliers to csv file
fl = "4. Analysis/used_car_prices_extreme_outliers.csv"
outliers.to_csv(fl, index=False)

print("Dataset: {}".format(df.shape))
df.drop( outliers.index, inplace=True )
df.reset_index(drop=True)
print('Drop outliers :{}'.format(len(outliers)))
print("Dataset: {}".format(df.shape))


###################################### Feature engineering ######################################

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
car_models = car_models[car_models['name'] < car_total].index
# create a new category'other' for rare car model names
df['name'] = df['name'].apply(lambda x: 'other' if x in car_models else x)

# Create features to reduce a number of categories
hybrid ='hybrid_or_electric'
df['engine_fuel'] = df['engine_fuel'].replace({'hybrid-petrol':hybrid,'hybrid-diesel':hybrid,'electric':hybrid})


# Create a list of unnamed features
features_list = ['feature_0','feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
# Count a total number of unnamed features for a car
df['other_features']=df[features_list].sum(axis=1)

# Round car prices
df['price_usd'] = df['price_usd'].apply(lambda x: round(x,-2))


###################################### Feature selection ######################################

# Define predictor and target variables
features =[ 'manufacturer_name', 'has_warranty', 'state', 'drivetrain', 'transmission', 'name',
           'odometer_value', 'odometer_value/year', 'year',  'engine_fuel','color',
           'duration_listed', 'body_type', 'engine_capacity', 'other_features', 'feature_0'
          ]

target = 'price_usd'

# Create the dataset to fit a model
data = df[features+ [target]].copy()
print('Dataset: {}'.format(data.shape))


###################################### Create pipeline ######################################

# Copy the initial dataset before transformation
df_origin = data.copy()

# Create features and target from the data set
X = data.drop(target ,axis=1)
y = data[target]

# Numeric data types
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Print numeric features
numeric_features = X.select_dtypes(numerics).columns.tolist()
print('Numeric features: {}'.format(numeric_features))


# Applies Power Transformer using Yeo-Johnson transformation to numeric columns 
numeric_power = ['odometer_value',  'odometer_value/year', 'duration_listed']

numeric_power_transformer = Pipeline(steps=
                                    [('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                     ('power', PowerTransformer(method='yeo-johnson')),
                                     ('scaler', StandardScaler())
                                    ])

# Applies Quantile Transformer to numeric columns 
numeric_quantile = ['engine_capacity', 'year', 'other_features']

numeric_quantile_transformer = Pipeline(steps=
                                    [('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                     ('quantile', QuantileTransformer(n_quantiles=100, output_distribution='normal')),
                                     ('scaler', StandardScaler())
                                    ])


# Print categorical features
categorical_features = X.select_dtypes([np.object,np.bool]).columns.tolist()
print('Categorical features: {}'.format(categorical_features))

# Transform categorical columns using OneHotEncoder
categorical_transformer = Pipeline(steps=
                                    [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                     ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                    ])
# Create ColumnTransformer to perform different transformations for different columns of the data
preprocessor = ColumnTransformer(transformers=
                                 [('num_power', numeric_power_transformer, numeric_power),
                                  ('num_qt', numeric_quantile_transformer, numeric_quantile),
                                  ('cat', categorical_transformer, categorical_features)
                                 ])

# Reshape target variable
y = np.array(y).reshape(-1,1)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Transform the target variable
power_tr = PowerTransformer(method='yeo-johnson')
y_train = power_tr.fit_transform(y_train)
y_test = power_tr.fit_transform(y_test)


###################################### Model ######################################

#################################### SVR model ####################################
# Create an instance of a model
model = SVR(C=1)
print(model)

# Define preprocessing pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', model)])  

y_train = y_train.ravel()
y_test = y_test.ravel()

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print('Score: {}'.format(score))

mse = cross_val_score(pipe, X_test, y_test, cv=10, scoring ='neg_mean_squared_error' ).mean()
print('MSE: {}'.format(mse))


###################################### Save the model ######################################
model_path = '5. Insights/Models/used_car_prices_model.pickle'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print('The file {} is saved.'.format(model_path))


###################################### Ptrediction ######################################

# Load the data
prediction_data = '2. Prepared Data/pred_cars.csv'
df = pd.read_csv(prediction_data)

# Load a model from a file
model_path = '5. Insights/Models/used_car_prices_model.pickle'
with open(model_path, 'rb') as f:
    regression_model_loaded = pickle.load(f)
print('Regression model {} is loaded.'.format(regression_model_loaded))


###################################### Preprocess the data ######################################

print(df.info())

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
car_models = car_models[car_models['name'] < car_total].index
# Create a new category'other' for rare car model names
df['name'] = df['name'].apply(lambda x: 'other' if x in car_models else x)

# Create features to reduce a number of categories
hybrid ='hybrid_or_electric'
df['engine_fuel'] = df['engine_fuel'].replace({'hybrid-petrol':hybrid,'hybrid-diesel':hybrid,'electric':hybrid})

# Create a list of unnamed features
features_list = ['feature_0','feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9']
# Count a total number of unnamed features for a car
df['other_features']=df[features_list].sum(axis=1)


###################################### Prediction of car prices ######################################

# Predict car prices from prediction file
model = regression_model_loaded

# Define preprocessing pipeline
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('model', model)])  
# Make prediction using the pipeline
prediction = pipe.predict(df[features])
prediction

# Transform predicted price to get the rounded car price in dollars
y_predict_price = power_tr.inverse_transform(prediction.reshape(-1,1))
y_predict_price_round = np.round(y_predict_price,-2)

# Form a dataframe with predicted results
results = pd.DataFrame( {'Predicted':y_predict_price.reshape(-1),
                         'Predicted rounded':y_predict_price_round.reshape(-1)},
                          index=df.index)
# Form a dataframe with car information and predicted prices
prediction_results = df.join(results)

# Save car information and predicted prices to csv file
fl = "5. Insights/Prediction/used_car_prices_predicted_price.csv"
prediction_results.to_csv(fl, index=False)




