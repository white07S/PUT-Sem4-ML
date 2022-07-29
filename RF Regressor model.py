
#Importing Libraries
import numpy as np
import pandas as pd
import hvplot.pandas
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

# %%
# Read APPL.csv contains open, high, low, close, Adj close, Volume of Apple stock with twitter polarity scores and twitter volume

df = pd.read_csv('AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)

# Drop null values
df.dropna(inplace=True)
df.tail()

# %%
# Dataframe with Adj close, ts_polarity, twitter_volume of APPL
df = df[["Adj Close", "ts_polarity", "twitter_volume"]]
df.head()

# %%
# pct change based on Adj close value
df["Pct_change"] = df["Adj Close"].pct_change()

# Drop null values
df.dropna(inplace = True)
df.head()


# %% [markdown]
# ### Creating the Features `X` and Target `y` Data
# 
# The first step towards preparing the data was to create the input features `X` and the target vector `y`. We used the `window_data()` function to create these vectors.
# 
# This function chunks the data up with a rolling window of _X<sub>t</sub> - window_ to predict _X<sub>t</sub>_.
# 
# The function returns two `numpy` arrays:
# 
# * `X`: The input features vectors.
# 
# * `y`: The target vector.
# 
# The function has the following parameters:
# 
# * `df`: The original DataFrame with the time series data.
# 
# * `window`: The window size in days of previous closing prices that will be used for the prediction.
# 
# * `feature_col_number`: The column number from the original DataFrame where the features are located.
# 
# * `target_col_number`: The column number from the original DataFrame where the target is located.

# %%
# This function "window_data" accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
    # Create empty lists "X_close", "X_polarity", "X_volume" and y
    X_close = []
    X_polarity = []
    X_volume = []
    y = []
    for i in range(len(df) - window):
        
        # Get close, ts_polarity, tw_vol, and target in the loop
        close = df.iloc[i:(i + window), feature_col_number1]
        ts_polarity = df.iloc[i:(i + window), feature_col_number2]
        tw_vol = df.iloc[i:(i + window), feature_col_number3]
        target = df.iloc[(i + window), target_col_number]
        
        # Append values in the lists
        X_close.append(close)
        X_polarity.append(ts_polarity)
        X_volume.append(tw_vol)
        y.append(target)
        
    return np.hstack((X_close,X_polarity,X_volume)), np.array(y).reshape(-1, 1)

# %%
# Predict Closing Prices using a 3 day window of previous closing prices
window_size = 3

# Column index 0 is the `Adj Close` column
# Column index 1 is the `ts_polarity` column
# Column index 2 is the `twitter_volume` column
feature_col_number1 = 0
feature_col_number2 = 1
feature_col_number3 = 2
target_col_number = 0
X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number)

# %%
# Use 70% of the data for training and the remainder for testing
X_split = int(0.7 * len(X))
y_split = int(0.7 * len(y))

X_train = X[: X_split]
X_test = X[X_split:]
y_train = y[: y_split]
y_test = y[y_split:]

# %% [markdown]
# # Scaling Data with `MinMaxScaler`
# 
# We will use the `MinMaxScaler` from `sklearn` to scale all values between `0` and `1`.
# Note that we scale both features and target sets.

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
# Use the MinMaxScaler to scale data between 0 and 1.
x_train_scaler = MinMaxScaler()
x_test_scaler = MinMaxScaler()
y_train_scaler = MinMaxScaler()
y_test_scaler = MinMaxScaler()

# Fit the scaler for the Training Data
x_train_scaler.fit(X_train)
y_train_scaler.fit(y_train)

# Scale the training data
X_train = x_train_scaler.transform(X_train)
y_train = y_train_scaler.transform(y_train)

# Fit the scaler for the Testing Data
x_test_scaler.fit(X_test)
y_test_scaler.fit(y_test)

# Scale the y_test data
X_test = x_test_scaler.transform(X_test)
y_test = y_test_scaler.transform(y_test)


# %%
# Create the Random Forest regressor instance
model = RandomForestRegressor(n_estimators=1000, max_depth=2, bootstrap=False, min_samples_leaf=1)


# %%
# Fit the model
model.fit(X_train, y_train.ravel())

# %% [markdown]
# ---

# %% [markdown]
# ---

# %% [markdown]
# ## Model Performance
# 
# In this section, we will evaluate the model using the test data. 
# 
# We will:
# 1. Evaluate the model using the `X_test` and `y_test` data.
# 2. Use the X_test data to make predictions
# 3. Create a DataFrame of Real (y_test) vs predicted values. 
# 4. Plot the Real vs predicted values as a line chart
# 

# %%
# Make some predictions
predicted = model.predict(X_test)

# %%
# Evaluating the model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R-squared :', metrics.r2_score(y_test, predicted))

# %%
# Recover the original prices instead of the scaled version
predicted_prices = y_test_scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))

# %%
# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
stocks.head()

# %%
# Plot the real vs predicted values as a line chart
stocks.hvplot(title = "Real vs Predicted values of APPL")

# %%


# %%



