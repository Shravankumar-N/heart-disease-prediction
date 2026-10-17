# ❤️ Heart Disease Prediction

An end-to-end Machine Learning project that predicts heart disease using Python, Scikit-learn, and Streamlit.

## 🚀 Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- One-Hot Encoding and feature scaling
- Multiple classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Hyperparameter tuning using GridSearchCV
- Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Interactive Streamlit web application

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 📂 Project Structure

```text
Heart_Disease/
├── app.py
├── Heart.ipynb
├── heart.csv
├── heart_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```
📊 Model Performance

Logistic Regression was selected for the web application based on the recorded experimental results.

| Metric | Score |
|---|---:|
| Accuracy | 89.13% |
| Precision | 89.42% |
| Recall | 91.18% |
| F1-score | 90.29% |

Results are based on the test dataset and do not represent guaranteed performance on new data.

⚙️ Installation and Usage

1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>

2. Navigate to the project folder

cd Heart_Disease

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python -m streamlit run app.py

The application will open in your browser.

🔮 Future Improvements
Improve model performance through cross-validation and tuning.
Add model interpretability and visualizations.
Deploy the application online.
⚠️ Disclaimer

This project is for educational purposes only. It is not a medically validated diagnostic tool and must not be used for medical decisions.

👨‍💻 Author

Developed as a Machine Learning project to explore classification, model evaluation, and web application development using Python.