import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔐 OpenRouter setup
import openai
from openai import OpenAI

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url=openai.api_base,
    api_key=openai.api_key
)

@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")
    df = df[df['House Style'].isin(['1Story', '2Story', 'SFoyer', 'SLvl'])]
    df = df[[
        'Gr Liv Area', 'Overall Qual', 'Year Built', 'Garage Cars',
        'Full Bath', 'Fireplaces', 'Total Bsmt SF', 'House Style',
        'Heating', 'Central Air', 'SalePrice'
    ]]
    df['Heating'] = df['Heating'].map({'GasA': 5, 'GasW': 4, 'Grav': 3, 'Wall': 2, 'Floor': 2, 'OthW': 1})
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df['House Style'] = df['House Style'].map({'1Story': 1, '2Story': 2, 'SFoyer': 1.5, 'SLvl': 1.7})
    df.dropna(inplace=True)
    return df

def train_and_predict(model_name, df, input_data):
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = (
        XGBRegressor(verbosity=0) if model_name == "XGBoost"
        else DecisionTreeRegressor() if model_name == "Decision Tree"
        else RandomForestRegressor()
    )
    model.fit(X_train, y_train)
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape

    return prediction[0], rmse, r2, accuracy

def main():
    st.set_page_config(page_title="Ames House Price Predictor", layout="centered")
    st.title("Ames Housing Price Prediction")
    df = load_data()

    for key in ["prediction", "rmse", "r2", "accuracy", "ai_explanation"]:
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.header("📋 Enter Input Features")
        gr_liv_area = st.number_input("Above-Ground Living Area (sq ft)", 500, 6000, 1500)
        overall_qual = st.slider("Overall Material Quality (1-10)", 1, 10, 5)
        year_built = st.number_input("Year Built", 1800, 2023, 2000)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
        full_bath = st.slider("Full Bathrooms", 0, 4, 2)
        fireplaces = st.slider("Fireplaces", 0, 3, 1)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
        heating = st.selectbox("Heating", ['GasA', 'GasW', 'Grav', 'Wall', 'Floor', 'OthW'])
        central_air = st.selectbox("Central Air", ['Yes', 'No'])
        house_style_display = st.selectbox("House Style", ['1 Floor', '2 Floors', 'Split Foyer', 'Split Level'])
        model_name = st.selectbox("Select Model", ["XGBoost", "Decision Tree", "Random Forest"])

        heating_encoded = {'GasA': 5, 'GasW': 4, 'Grav': 3, 'Wall': 2, 'Floor': 2, 'OthW': 1}[heating]
        central_air_encoded = 1 if central_air == 'Yes' else 0
        house_style_encoded = {'1 Floor': 1, '2 Floors': 2, 'Split Foyer': 1.5, 'Split Level': 1.7}[house_style_display]

        input_data = [
            gr_liv_area, overall_qual, year_built, garage_cars,
            full_bath, fireplaces, total_bsmt_sf,
            house_style_encoded, heating_encoded, central_air_encoded
        ]

    if st.button("Predict Sale Price"):
        prediction, rmse, r2, accuracy = train_and_predict(model_name, df, input_data)
        st.session_state.prediction = prediction
        st.session_state.rmse = rmse
        st.session_state.r2 = r2
        st.session_state.accuracy = accuracy
        st.session_state.ai_explanation = None

    if st.session_state.prediction is not None:
        st.success(f"🏷️ Predicted Sale Price: ${st.session_state.prediction:,.2f}")
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 RMSE", f"${st.session_state.rmse:,.2f}")
        col2.metric("📈 R² Score", f"{st.session_state.r2:.2f}")
        col3.metric("📏 Accuracy", f"{st.session_state.accuracy:.2f}%")

        st.subheader("Where Does Your Price Fall?")
        fig, ax = plt.subplots()
        sns.histplot(df['SalePrice'], bins=30, kde=True, color="skyblue", ax=ax)
        ax.axvline(st.session_state.prediction, color='red', linestyle='--',
                   label=f'Your Predicted Price: ${st.session_state.prediction:,.0f}')
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Count of Homes")
        ax.set_title("Distribution of Ames Home Sale Prices")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Want AI to Explain This?")
    if st.button("Explain with AI"):
        if st.session_state.prediction is not None:
            prompt = f"""
            A house in Ames, Iowa was predicted to be worth ${st.session_state.prediction:,.0f}.
            The RMSE of the model is ${st.session_state.rmse:,.0f}, the R² score is {st.session_state.r2:.2f},
            and the average prediction accuracy is {st.session_state.accuracy:.2f}%.
            The predicted price falls within the peak of the historical price distribution.
            Please explain this in clear terms for a homebuyer or homeowner with no machine learning background.
            """
            with st.spinner("Asking AI via OpenRouter..."):
                try:
                    response = client.chat.completions.create(
                        model="mistralai/mistral-7b-instruct",
                        messages=[
                            {"role": "system", "content": "You are a helpful real estate data analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.7
                    )
                    st.session_state.ai_explanation = response.choices[0].message.content
                except Exception as e:
                    st.error(f"AI request failed: {e}")
        else:
            st.warning("Please predict the sale price first.")

    if st.session_state.ai_explanation:
        st.markdown("**🧠 AI Explanation:**", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                padding: 1rem 1.5rem;
                border-radius: 10px;
                border: 1px solid #444;
                font-family: 'Segoe UI', sans-serif;
                overflow-wrap: break-word;
                white-space: pre-wrap;
                line-height: 1.6;
            ">
            {st.session_state.ai_explanation.replace('\n', '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing before expander

    with st.expander("Explore the Data"):
        st.write("Here's a preview of the dataset:")
        st.dataframe(df.head())
        st.write("Correlation Heatmap:")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
