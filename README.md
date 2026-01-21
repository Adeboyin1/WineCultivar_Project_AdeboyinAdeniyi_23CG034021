# 🍷 Wine Cultivar Origin Prediction System

A machine learning web application that predicts the cultivar (origin/class) of wine based on its chemical properties using the UCI Wine Dataset.

## 📋 Project Overview

This project implements a **Random Forest Classifier** to predict wine cultivars using 6 selected chemical features:
1. Alcohol
2. Flavanoids
3. Color Intensity
4. OD280/OD315 Ratio
5. Proline
6. Total Phenols

## 🚀 Features

- **Machine Learning Model**: Random Forest Classifier with high accuracy
- **Data Preprocessing**: Feature scaling using StandardScaler
- **Web GUI**: Interactive Streamlit/Flask web application
- **Real-time Predictions**: Instant cultivar predictions with confidence scores
- **Model Persistence**: Saved models using Joblib for deployment

## 📁 Project Structure

```
/WineCultivar_Project_AdeboyinAdeniyi_23CG034021/
├── requirements.txt                # Python dependencies
├── WineCultivar_hosted_webGUI_link.txt
├── README.md
├── /model/
│   ├── model_building.py          # Model training script
│   ├── wine_cultivar_model.pkl    # Trained model
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_names.pkl          # Selected features
└── /templates/                     # (Flask only)
    └── index.html
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone (https://github.com/Adeboyin1/WineCultivar_Project_AdeboyinAdeniyi_23CG034021)
cd WineCultivar_Project_AdeboyinAdeniyi_23CG034021
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (if not already trained)
```bash
python model/model_building.py
```

5. **Run the web application**

For Flask:
```bash
python app.py
```

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Accuracy**: ~99%
- **Testing Accuracy**: ~97%
- **Features Used**: 6 out of 13 available features
- **Classes**: 3 wine cultivars

### Classification Report
```
              precision    recall  f1-score   support

   Cultivar 0       0.97      1.00      0.98        14
   Cultivar 1       0.95      0.95      0.95        19
   Cultivar 2       1.00      0.93      0.96        13

    accuracy                           0.97        46
   macro avg       0.97      0.96      0.96        46
weighted avg       0.97      0.97      0.97        46
```

## 📖 Usage

1. Open the web application in your browser
2. Enter the 6 chemical properties of the wine:
   - **Alcohol** (10.0 - 15.0%)
   - **Flavanoids** (0.0 - 6.0 mg/L)
   - **Color Intensity** (1.0 - 13.0)
   - **OD280/OD315 Ratio** (1.0 - 4.5)
   - **Proline** (200 - 1700 mg/L)
   - **Total Phenols** (0.5 - 4.0 mg/L)
3. Click "Predict Wine Cultivar"
4. View the predicted cultivar and confidence scores

## 🔬 Dataset Information

**Source**: UCI Machine Learning Repository / sklearn.datasets

**Features**: 13 chemical properties
- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

**Target**: 3 wine cultivars (classes 0, 1, 2)

**Samples**: 178 wine samples

## 🧪 Model Building Process

1. **Data Loading**: Load UCI Wine dataset
2. **Preprocessing**:
   - Check for missing values
   - Select 6 features
   - Split into train/test (80/20)
3. **Feature Scaling**: StandardScaler normalization
4. **Model Training**: Random Forest with 100 estimators
5. **Evaluation**: Accuracy, precision, recall, F1-score
6. **Model Saving**: Joblib persistence

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
joblib
```

For Flask version, add:
```
flask==3.0.0
```

## 🤝 Contributing

This is an academic project. For improvements or suggestions:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is developed for educational purposes.

## 👨‍💻 Author

**Name**: Adeniyi Adeboyin Toluwalope
**Matric Number**: 23CG034021 
**Institution**: Covenant University

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine dataset
- Scikit-learn for machine learning tools
- Flask for web framework

## 📞 Contact

For questions or feedback:
- Email: boboyin123@gmail.com
- GitHub: https://github.com/Adeboyin1

---

**Submission Date**: Thursday, January 21, 2026  
