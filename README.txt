
Files included:
- requirements.txt : Python packages to install (use pip install -r requirements.txt)
- feature_info.json : expected feature names and notes
- train_model.py : script to run in Google Colab to train a RandomForest on your Attrition.csv and save model.pkl
- streamlit_app.py : Streamlit app that loads model.pkl and feature_info.json and provides an interactive UI
- model.pkl : placeholder model shipped with this repo (trained on synthetic data) - you should replace it by running train_model.py on your real dataset

How to use (Google Colab -> GitHub -> Streamlit):
1) In Google Colab: mount your Drive and run train_model.py after updating DATA_PATH to point to your Attrition.csv.
   Example:
     from google.colab import drive
     drive.mount('/content/drive')
     !python train_model.py
   This will produce model.pkl (containing {'model': model, 'columns': column_list}).
2) Push files to your GitHub repo (train_model.py, streamlit_app.py, model.pkl, feature_info.json, requirements.txt)
3) Deploy to Streamlit Cloud or run locally with: streamlit run streamlit_app.py
4) If the app complains about missing columns, open feature_info.json and adjust 'features' to match your training dataset column order.

Notes:
- The provided model.pkl is a placeholder trained on synthetic data so the app can run out-of-the-box for demo purposes.
- For best results, run train_model.py with your real Attrition.csv to create a production-ready model.pkl.
