
# Initial imports

import pandas as pd
import numpy as np
from path import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# %% [markdown]
# ## Loading and Preprocessing Data
# 
# Load the `APPL.csv` in a pandas DataFrame called `df`

# %%
# Loading data
file_path = Path("AAPL.csv")
df = pd.read_csv(file_path)
df.head()


# %%
# Dataframe with Date, Adj close,  Volume, ts_polarity, twitter_volume of APPL
appl_df = df[["Date", "Adj Close", "Volume", "ts_polarity", "twitter_volume"]]
appl_df.head()

# %%
# Setting Index as Date
appl_df = appl_df.dropna()
appl_df.set_index("Date", inplace = True)
appl_df.tail()

# %%
# Sorting ts_polarity into Positive, Negative and Neutral sentiment

sentiment = [] 
for score in appl_df['ts_polarity']:
    if score >= 0.05 :
          sentiment.append("Positive") 
    elif score <= - 0.05 : 
          sentiment.append("Negative")        
    else : 
        sentiment.append("Neutral")   

appl_df["Sentiment"] = sentiment
appl_df.head()

# %%
# Sentiment Count
appl_df['Sentiment'].value_counts()

# %%
#Stock Trend based on difference between current price to previous day price and coverting them to '0' as fall and '1' as rise in stock price
appl_df['Price Diff'] = appl_df['Adj Close'].diff()
appl_df.dropna(inplace = True)
appl_df['Trend'] = np.where(
    appl_df['Price Diff'] > 0 , 1, 0)

appl_df.head()

# %%
# Binary encoding Sentiment column
appl_trend = appl_df[["Adj Close", "Volume", 'twitter_volume', "Sentiment", "Trend"]]
appl_trend = pd.get_dummies(appl_trend, columns=["Sentiment"])
appl_trend.head()

# %%
# Defining features set
X = appl_trend.copy()
X.drop("Trend", axis=1, inplace=True)
X.head()


# %%
# Defining target vector
y = appl_trend["Trend"].values.reshape(-1, 1)
y[:5]


# %% [markdown]
# Split the data into training and testing sets.

# %%
# Splitting into Train and Test sets
split = int(0.7 * len(X))

X_train = X[: split]
X_test = X[split:]

y_train = y[: split]
y_test = y[split:]

# %%
# Using StandardScaler to scale features data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# %% [markdown]
#  ## Create a Random Forest Classifier Model
# 

# %%
# Create RFClassifier model
rf_model = RandomForestClassifier(n_estimators=500, random_state=78)

# Fit the model
rf_model = rf_model.fit(X_train_scaled, y_train.ravel())  


# %% [markdown]
# ## Making Predictions Using the Random Forest Classifier Model

# %%
# Make predictions
predictions = rf_model.predict(X_test_scaled)
pd.DataFrame({"Prediction": predictions, "Actual": y_test.ravel()}).head(20)

# Generate accuracy score for predictions using y_test
acc_score = accuracy_score(y_test, predictions)
print(f"Accuracy Score : {acc_score}")

# %% [markdown]
# ## Model Evaluation
# 
# Evaluating model's results, using `sklearn` to calculate the confusion matrix and to generate the classification report.

# %%
# Generating the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

# Displaying results
display(cm_df)

# %%
# Generating classification report
print("Classification Report")
print(classification_report(y_test, predictions))


# %%



