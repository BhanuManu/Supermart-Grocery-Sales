# Supermart Grocery Sales Analytics

This project is a web application that predicts grocery sales using a combination of a Random Forest Regression model and an Artificial Neural Network (ANN). The prediction is based on features such as Category, Sub-Category, City, Region, Year, Month, Discount, and Profit.

## Directory Structure

The project has been organized for better maintainability:

```
Supermart-Grocery-Sales/
│
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
│
├── data/                 # Directory containing datasets
│   └── Supermart Grocery Sales - Retail Analytics Dataset.csv
│
├── models/               # Directory containing trained models and encoders
│   ├── ann_model.keras
│   ├── Random_Forest_Regression.pkl
│   ├── catenc.pkl
│   ├── cityenc.pkl
│   ├── monthenc.pkl
│   ├── regenc.pkl
│   ├── subenc.pkl
│   └── yrenc.pkl
│
└── notebooks/            # Directory containing Jupyter notebooks for data analysis and model training
    └── supermarket (1).ipynb
```

## Setup and Installation

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the web interface at `http://127.0.0.1:5000/`.

## Technologies Used
- Python
- Flask
- TensorFlow (Keras)
- scikit-learn (Joblib for models/encoders)
- Pandas & NumPy
