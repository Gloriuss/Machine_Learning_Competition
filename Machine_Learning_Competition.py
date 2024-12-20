# Imports
# GENERAL
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import os

# Data Download
ytest_path = os.path.join(os.path.dirname(__file__), 'Data_Download/SubmissionFormat.csv')
xtest_path = os.path.join(os.path.dirname(__file__), 'Data_Download/test_set_values.csv')
ytrain_path = os.path.join(os.path.dirname(__file__), 'Data_Download/training_set_labels.csv')
xtrain_path = os.path.join(os.path.dirname(__file__), 'Data_Download/training_set_values.csv')

#Data in variables
ytest = pd.read_csv(ytest_path)
xtest = pd.read_csv(xtest_path)
ytrain = pd.read_csv(ytrain_path)
xtrain = pd.read_csv(xtrain_path)

# Parameterization and function definition
pd.set_option('display.max_columns', None)

def convertir_a_numerico(fila):
    # Assign 0 to NaN values
    fila = fila.fillna(0)
    
    valores_str = fila.astype(str)  # Convert all values to strings
    valor_numerico = {valor: i for i, valor in enumerate(sorted(valores_str.unique()))}
    fila_numerica = fila.map(valor_numerico)
    
    # Revert 0 back to NaN after conversion
    fila_numerica = fila_numerica.replace(0, np.nan)
    
    return fila_numerica

# Variable Definitions
# Concatenate train and test for better KNN imputation
df_train = xtrain
df_train['origen_data'] = 1
df_test = xtest
df_test['origen_data'] = 2
df = pd.concat([df_train, df_test], ignore_index=True)

# Identify data origins
df1 = df[df['origen_data'] == 1]
df2 = df[df['origen_data'] == 2]

# TARGET VARIABLE
name_varobj = {'status_group': 'var_obj'}
ytrain = ytrain.rename(columns=name_varobj)

# Assign values to the categorical target variable
cat_var_obj = {valor: i for i, valor in enumerate(sorted(ytrain['var_obj'].unique()))}
print(cat_var_obj)
ytrain['var_obj'] = ytrain['var_obj'].map(cat_var_obj)
VarObj = ytrain['var_obj']

# Data Cleaning
## Remove columns with only 1 unique value
# Calculate the number of unique values in each column
valores_unicos_por_columna = df.nunique()

# Filter columns with only 1 unique value
columnas_con_un_valor_unico = valores_unicos_por_columna[valores_unicos_por_columna == 1].index

# Drop the selected columns from the original DataFrame
df = df.drop(columns=columnas_con_un_valor_unico)

## Booleans
df['public_meeting'] = df['public_meeting'].map({True: 1, False: 0})
df['permit'] = df['permit'].map({True: 1, False: 0})

# Iterate over boolean columns
for columna in df.select_dtypes(include=bool):
    # Fill NaN with 0, then map False to 1 and True to 2
    df[columna] = df[columna].fillna(0).astype(bool).map({False: 1, True: 2})

## Dates
# Consider as dates
df['date_recorded'] = pd.to_datetime(df['date_recorded'])

# Calculate the maximum date
max_date_rec = df['date_recorded'].max()

# Calculate the difference in days for each date compared to 'max_date_rec'
df['date_recorded'] = (max_date_rec - df['date_recorded']).dt.days

# Adjust the maximum date to 0 and other dates based on the day difference
df['date_recorded'] = df['date_recorded'].max() - df['date_recorded']

## Identify variable types in the DataFrame
columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns
columnas_numericas = df.select_dtypes(include=['float', 'int']).columns

## Convert categories to numbers
# Apply the function to each column in the DataFrame
for columna in df.columns:
    if df[columna].dtype == 'object' or df[columna].dtype.name == 'category':
        df[columna] = convertir_a_numerico(df[columna])

# Imputation
# Filter columns by data type
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create a new DataFrame with only numeric columns
df = df[numeric_columns]

# Create an imputer using the most frequent strategy (mode)
imputerknn = KNNImputer(n_neighbors=4, weights="uniform")

# Iterate over each column in the DataFrame and apply imputation
for column in df.columns:
    # Check if the column contains numeric, object, or category values
    if df[column].dtype in ['int64', 'float64']:
        # Impute NaN values in the current column
        df[column] = imputerknn.fit_transform(df[[column]]).ravel()

# Train Test Split
# Redefine train and test DataFrames based on data handling
df1 = df[df['origen_data'] == 1]
df2 = df[df['origen_data'] == 2]
x_train = df1
y_train = ytrain['var_obj']

# Model
# 1. Model selection
model = RandomForestClassifier()
print(type(model))

# 2. Train model
model.fit(x_train, y_train)

# 3. Predict
y_model = model.predict(df2)
print(y_model)

y_model_df = pd.DataFrame({'var_obj': y_model})
y_model_df['var_obj'] = y_model_df['var_obj'].map({v: k for k, v in cat_var_obj.items()})
prediccion = pd.merge(ytest['id'], y_model_df, left_index=True, right_index=True, suffixes=('_ytest', '_y_model'))
prediccion.rename(columns={'var_obj': 'status_group'}, inplace=True)
prediccion.to_csv('Prediction_Result/y_model_predictions.csv', index=False)