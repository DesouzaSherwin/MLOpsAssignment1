import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('housing.csv')


X = df.drop('median_house_value',axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


imputer = SimpleImputer(strategy='median')
X_train['total_bedrooms'] = imputer.fit_transform(X_train[['total_bedrooms']])


X_test['total_bedrooms'] = imputer.transform(X_test[['total_bedrooms']])



encoder = OneHotEncoder(handle_unknown='ignore')


ocean_proximity_train_encoded = encoder.fit_transform(X_train[['ocean_proximity']])

ocean_proximity_test_encoded = encoder.transform(X_test[['ocean_proximity']])

ocean_proximity_train_df = pd.DataFrame(ocean_proximity_train_encoded.toarray(), 
                                        columns=encoder.get_feature_names_out(['ocean_proximity']))
ocean_proximity_test_df = pd.DataFrame(ocean_proximity_test_encoded.toarray(), 
                                       columns=encoder.get_feature_names_out(['ocean_proximity']))

X_train = pd.concat([X_train.reset_index(drop=True), ocean_proximity_train_df], axis=1).drop('ocean_proximity', axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), ocean_proximity_test_df], axis=1).drop('ocean_proximity', axis=1)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2}")

print(encoder.categories_)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)



with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('imputer.pkl', 'wb') as imputer_file:
    pickle.dump(imputer, imputer_file)

with open('encoder.pkl', 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)    