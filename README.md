# Car Prices Prediction

The project is created to automate process of estimation of used car price based on set of its features.  It is supposed to be used by a company that specialized in selling used cars. The dataset was provided by SuperDataScience platform.  

A nonlinear regression model was built using Support Vector Regression (SVR) with 'rbf' kernel. Mean squared error (𝑀𝑆𝐸) and coefficient of determination (R2 score) were used as the performance metrics. The result model R2 score was around of 91%.  

#### Contents of files

[Car_Prices_Regression_EDA.ipynb](Car_Prices_Regression_EDA_v5.ipynb) file contains Exploratory Data Analysis of the data.  
[Car_Prices_Regression_Modeling.ipynb](Car_Prices_Regression_Modeling_v6.ipynb) file contains model selection and tuning, model training using preprocessing pipeline, saving a customer feature transformer class in a file (['feature_transformation.py'](feature_transformation.py)), saving the model in a pickle file '5. Insights/Models/used_car_prices_model.pickle'.  
[Car_Prices_Regression_MODEL_USAGE_EXAMPLE.ipynb](Car_Prices_Regression_MODEL_USAGE_EXAMPLE.ipynb) file contains an example of using the model for prediction a car price.  
['project_description.doc'](project_description.doc) file contains the project description in more details.






Data source: [https://github.com/edis/sds_challenges/tree/master/challenge_2/data](https://github.com/edis/sds_challenges/tree/master/challenge_2/data).
