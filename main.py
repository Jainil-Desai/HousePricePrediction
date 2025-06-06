import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def generate_house_data(num_houses=100):
    np.random.seed(50)
    size = np.random.normal(1400, 50, num_houses)
    price = size * 50 + np.random.normal(0, 50, num_houses)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    df = generate_house_data(num_houses=100)
    X = df[['size']]
    Y = df['price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, Y_train)

    return model

def main():
    st.title("House Price Prediction")
    st.write("This app predicts the price of a house based on its size.")

    model = train_model()

    size = st.number_input("Enter the size of the house (in square feet)", min_value=0, max_value=2000, value = 1500)

    if st.button("Predict"):
        prediction = model.predict([[size]])
        st.write("Predicted price:", prediction[0])

        df  = generate_house_data()

        fig = px.scatter(df, x = 'size', y = 'price')
        fig.add_scatter(x = [size], y = [prediction[0]], mode = 'markers', name = 'Predicted Price')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()