
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Edit this path in Colab to point to your Attrition.csv (for example: '/content/drive/MyDrive/PPA/Attrition.csv')
DATA_PATH = "/content/drive/MyDrive/PPA/Attrition.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # Drop columns not needed if present
    for c in ['Over18','EmployeeCount','StandardHours']:
        if c in df.columns:
            try:
                df.drop(c, axis=1, inplace=True)
            except:
                pass
    # Map binaries if present
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
    if 'OverTime' in df.columns:
        df['OverTime'] = df['OverTime'].map({'Yes':1,'No':0})

    # One-hot encode categorical columns commonly used in the notebook
    categorical_cols = [c for c in ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'] if c in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    # Fill NA with median for numeric cols
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    return df

def train_and_save(df, model_path='model.pkl'):
    # target must be Attrition
    if 'Attrition' not in df.columns:
        raise ValueError("Dataset must contain 'Attrition' column with Yes/No values.")
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    # save model
    with open(model_path, 'wb') as f:
        pickle.dump({'model': rf, 'columns': X.columns.tolist()}, f)
    print(f"Model trained and saved to {model_path}. Columns saved: {len(X.columns)}")

if __name__ == '__main__':
    print("This script trains a RandomForest model on Attrition.csv and saves model.pkl.")
    print("Edit DATA_PATH variable to point to your file in Colab (eg: '/content/drive/MyDrive/PPA/Attrition.csv'), then run this script.")
