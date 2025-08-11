#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt


# ### Load datasets

# In[ ]:


from sklearn.datasets import fetch_california_housing
df_housing = fetch_california_housing()


# In[ ]:


type(df_housing)


# In[ ]:


df_housing.feature_names


# In[ ]:


df_housing.data


# In[ ]:


df_housing.keys()


# In[ ]:


print(df_housing.DESCR)


# In[ ]:


# Assuming df_housing is a dictionary with nested structures

# Improved flattening function to ensure 1D arrays
def flatten_value(val):
    if isinstance(val, (list, np.ndarray)):
        # If it's a multi-dimensional array/list, flatten it to a single value
        if len(val) > 0:
            # Take first element if it exists
            return flatten_value(val[0])  # Recursively flatten if needed
        else:
            return None
    return val

# First, let's properly flatten the data structure
flattened_data = {}
for key, value in df_housing.items():
    if isinstance(value, (list, np.ndarray)):
        # Make sure all values in the list are scalar
        flattened_data[key] = [flatten_value(item) for item in value]
    else:
        flattened_data[key] = value

# Ensure all lists are of the same length
max_length = max(len(val) if isinstance(val, list) else 1 for val in flattened_data.values())
for key, value in flattened_data.items():
    if not isinstance(value, list):
        flattened_data[key] = [value] * max_length
    elif len(value) < max_length:
        flattened_data[key] = value + [None] * (max_length - len(value))

# Create DataFrame from the flattened dictionary
df = pd.DataFrame(flattened_data)

# Alternative approach if df_housing is already a DataFrame:
# df = pd.DataFrame()
# for col in df_housing.columns:
#     # Extract and flatten data from each column
#     df[col] = df_housing[col].apply(lambda x: flatten_value(x))


# ## df 

# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


df_housing.data


# In[ ]:


df = pd.DataFrame(df_housing.data,columns= df_housing.feature_names)


# In[ ]:


df


# In[ ]:


df_housing.target


# In[ ]:


df['Price'] = df_housing.target


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()


# ## EDA

# In[ ]:


## Correaltion

df.corr()


# In[ ]:


import seaborn as sns

sns.pairplot(df)


# In[ ]:


plt.scatter(df['HouseAge'],df['Price'])
plt.xlabel('HouseAge')
plt.ylabel('Price')


# In[ ]:


df.columns


# In[ ]:


plt.scatter(df['Population'],df['Price'])
plt.xlabel('Population')
plt.ylabel('Price')
plt.scatter(df['MedInc'],df['Price'])
plt.xlabel('MedInc')
plt.ylabel('Price')
plt.scatter(df['AveRooms'],df['Price'])
plt.xlabel('AveRooms')
plt.ylabel('Price')
plt.scatter(df['AveBedrms'],df['Price'])
plt.xlabel('AveBedrms')
plt.ylabel('Price')
plt.scatter(df['AveOccup'],df['Price'])
plt.xlabel('AveOccup')
plt.ylabel('Price')
plt.scatter(df['Latitude'],df['Price'])
plt.xlabel('Latitude')
plt.ylabel('Price')
plt.scatter(df['Longitude'],df['Price'])
plt.xlabel('Longitude')
plt.ylabel('Price')


# In[ ]:


sns.regplot(x='Longitude',y='Price',data= df)


# In[ ]:


sns.regplot(x='Population',y= 'Price',data = df)


# In[ ]:


#MedInc
sns.regplot(x='MedInc',y= 'Price',data = df)


# ## Dependant and independent features

# In[ ]:


x = df.iloc[:,:-1]


# In[ ]:


x


# In[ ]:


y = df['Price']


# In[ ]:


print(x)
print(y)


# ## Train & test split

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)


# In[ ]:


x_train


# In[ ]:


x_test


# ## Standard scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


x_train = scaler.fit_transform(x_train)
print(x_train)
x_test = scaler.transform(x_test)
print(x_test)


# ## Model Training

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)


# In[ ]:


## Print the coifficients and intercepts
print(df.columns)
print(lin_reg.coef_)


# In[ ]:


print(lin_reg.intercept_)


# In[ ]:


lin_reg.get_params()  ## to get parameters of algorithm


# In[ ]:


# prediction with test data 

x_pred_test =  lin_reg.predict(x_test)
x_pred_test


# In[ ]:


print(x_pred_test)


# In[ ]:


# plot for pred and test values 
plt.scatter(y_test, x_pred_test)
plt.xlabel('y_test')
plt.ylabel('x_pred_test')


# In[ ]:


## find out the residuals

residuals = y_test - x_pred_test
print(residuals)


# In[ ]:


# plot the rsiduals 
sns.displot(residuals,kind='kde')


# In[ ]:


## plot the scatter with respect to predictions and residuals 

plt.scatter(x_pred_test,residuals)
plt.xlabel('x_pred_test')
plt.ylabel('residuals')   ## uniform distributed data 


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


mse = mean_squared_error(y_test,x_pred_test)
mae = mean_absolute_error(y_test,x_pred_test)
print(mae)
print(mse)
rmse = np.sqrt(mse)
print(rmse)


# In[ ]:


from sklearn.metrics import r2_score

score = r2_score(y_test,x_pred_test)
print(score)


# In[ ]:


## df_housing
df_housing.data[0]


# In[ ]:


df


# In[ ]:


lin_reg.predict(scaler.transform(df_housing.data[0].reshape(1,-1)))


# In[ ]:


import pickle

pickle.dump(lin_reg,open('lin_reg_model.pkl','wb'))
pickle_model = pickle.load(open('lin_reg_model.pkl','rb'))
print(pickle_model)


# In[ ]:


pickle_model.predict(scaler.transform(df_housing.data[0].reshape(1,-1)))


# In[ ]:


row = df.loc[1]         # or df.iloc[1]
json_obj = row.to_json()

