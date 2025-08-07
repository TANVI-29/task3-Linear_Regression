# Insurance Cost Prediction

This project predicts medical insurance charges based on patient details like age, BMI, number of children, gender, smoking status, and region using a Linear Regression model. The dataset contains the following columns: age, sex, bmi, children, smoker, region, and charges (target variable). We first drop the original categorical columns (`sex`, `smoker`, `region`) since we will encode them separately. We apply OneHotEncoding to transform these categorical columns into numerical format. Then we combine the encoded columns with the numerical data. Here's the preprocessing code used:

```python
df_cleaned = df.drop(["sex", "smoker", "region"], axis=1)

oe = OneHotEncoder(dtype=np.int32, sparse_output=False)
encoded = oe.fit_transform(df[["sex", "smoker", "region"]])

encoded_df = pd.DataFrame(encoded, columns=oe.get_feature_names_out(["sex", "smoker", "region"]))
final_df = pd.concat([df_cleaned, encoded_df], axis=1)
```

We then split the data into training and testing sets using an 80-20 ratio and train a Linear Regression model.

```python
x = final_df.drop("charges", axis=1)
y = final_df["charges"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
```

After training, we evaluate the model's performance using R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).

```python
y_pred = model.predict(x_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
```

We also print the intercept and coefficients to understand how each feature contributes to the prediction:

```python
print("Intercept:", model.intercept_)
for i in range(len(x_train.columns)):
    print(x_train.columns[i], ":", round(model.coef_[i], 2))
```

The linear regression model provides a simple and interpretable baseline for predicting medical insurance charges. It highlights which patient factors have the most influence on costs. To improve prediction accuracy, more advanced models like Random Forest, Gradient Boosting, or Neural Networks can be used in the future.