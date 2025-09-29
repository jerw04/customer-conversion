# Customer Conversion Prediction (Clickstream Analysis)

This project predicts whether a user session will lead to a customer making a purchase (conversion) using clickstream data from an online shopping dataset.

**Live App:**  
[Streamlit App – Predict Customer Conversion](https://jerw04-customer-conversion-appstreamlit-app-x7sbzy.streamlit.app/)

---

## 🚀 Project Structure

customer-conversion/
│
├── notebooks/
│ └── preprocessing.ipynb ← Data cleaning, feature engineering, model training & evaluation
│
├── data/
│ └── e-shop clothing 2008.csv ← Original UCI dataset
│
├── rf_model.pkl ← Saved Random Forest model used in the app
│
└── app/
└── streamlit_app.py ← Streamlit application for predictions & visualizations

## 🧩 How It Works

1. **Data Preprocessing & Feature Engineering**  
   - Load raw clickstream data  
   - Clean / rename columns, derive `converted` target, aggregate per session  
   - Create features like average price, unique models viewed, last page reached, etc.

2. **Model Training & Evaluation**  
   - Baseline model: Logistic Regression  
   - Advanced models: Random Forest & XGBoost  
   - Model comparison based on accuracy, F1, ROC-AUC  
   - Final model saved (`rf_model.pkl`) for deployment

3. **Streamlit Application**  
   - Users input session-level features  
   - Model predicts if the session likely converts (purchase) or not  
   - Probability score shown  

4. **Deployment**  
   - App hosted on Streamlit Cloud  
   - Live URL above  

---

## 🛠 Instructions to Run Locally

1. Clone this repository:

```bash
git clone <your-repo-url>
cd customer-conversion
```
2. Set up Python environment (e.g. using venv or conda) and install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```
4. The app will open in your browser at : http://localhost:8501
