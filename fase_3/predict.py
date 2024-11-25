import os
import pickle

import pandas as pd
import tensorflow_decision_forests as tfdf

from train import preprocess, tokenize_names

project_dir = os.path.join(os.path.dirname(__file__))
serving_data_path = os.path.join(project_dir, "test.csv")

serving_df = pd.read_csv(serving_data_path)

preprocessed_serving_df = preprocess(serving_df)

serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)

model_pkl_file = "model.pkl"

# load model from pickle file
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)

def prediction_to_kaggle_format(model, threshold=0.5):
    proba_survive = model.predict(serving_ds, verbose=0)[:,0]  # Predice las probabilidades de supervivencia para el conjunto de prueba
    return pd.DataFrame({
        "PassengerId": serving_df["PassengerId"],  # Incluye el identificador del pasajero
        "Survived": (proba_survive >= threshold).astype(int)  # Convierte las probabilidades en etiquetas binarias usando el umbral
    })

predictions = prediction_to_kaggle_format(model)

