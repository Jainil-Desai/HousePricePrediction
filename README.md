# Housing Price Predictor

A machine learning-powered web app built with **Streamlit** that predicts house prices in Ames, Iowa using input features such as square footage, quality, year built, and more. It also includes an optional **AI explanation** powered by OpenRouter's LLMs to help users understand the model’s predictions in plain English.

---

## Features

- Predict house prices using:
  - XGBoost (default)
  - Random Forest
  - Decision Tree
- Model metrics:
  - RMSE (Root Mean Squared Error)
  - R² Score (Goodness of fit)
  - Estimated Accuracy (based on MAPE)
- Visualization:
  - Distribution plot with your predicted price overlayed
- AI Explanation:
  - Uses OpenRouter API to explain predictions in simple language

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- OpenRouter (for LLM integration)
- Matplotlib & Seaborn for plotting

---

## File Structure

```
.
├── AmesHousing.csv              # Dataset
├── app.py                       # Main Streamlit app
├── requirements.txt             # Python dependencies
├── .streamlit/
│   └── secrets.toml             # API Key (not committed to Git)
└── .gitignore                   # Prevents secrets and data from being pushed
```

---

## Setup (API Key Security)

Store your OpenRouter API key in `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

This keeps your key secure and prevents accidental exposure when pushing to GitHub.

---

## Installation & Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/ames-price-predictor.git
cd ames-price-predictor
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Add your OpenRouter API key:**

Create a `.streamlit/secrets.toml` file as shown above.

4. **Run the app:**

```bash
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub.
3. Set the `OPENROUTER_API_KEY` in the Streamlit Cloud **Secrets Manager**.
4. Deploy and share!
