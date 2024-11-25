import os
import pickle

import pandas as pd
import tensorflow_decision_forests as tfdf

from typing import Annotated
from fastapi import FastAPI, Body
from train import preprocess, tokenize_names
from model_features import ModelParams

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": True})

global model_pkl_file
model_pkl_file = "model.pkl"

@app.get("/")
async def root():
    return {"message": "Probability of surviving the titanic ML API"}


@app.get("/train")
async def train():
    print('Training started')

    project_dir = os.path.join(os.path.dirname(__file__))
    train_data_path = os.path.join(project_dir, "train.csv")
    train_df = pd.read_csv(train_data_path)
    preprocessed_train_df = preprocess(train_df)
    input_features = list(preprocessed_train_df.columns)

    # Elimina la columna 'Ticket', ya que no se utilizará directamente como característica para el modelo.
    input_features.remove("Ticket")

    # Elimina la columna 'PassengerId', ya que es un identificador único que no aporta valor predictivo.
    input_features.remove("PassengerId")

    # Elimina la columna 'Survived', ya que es la etiqueta objetivo (lo que queremos predecir).
    input_features.remove("Survived")
    
    # Imprime las características de entrada seleccionadas que serán utilizadas por el modelo.
    print(f"Características de entrada: {input_features}")

    print("The data frame have been preprocesed!")
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df, label="Survived").map(tokenize_names)

    # Define un modelo de Gradient Boosted Trees usando TensorFlow Decision Forests.
    # Configura el modelo con los siguientes parámetros:
    # - verbose=0: Muestra muy pocos logs durante el entrenamiento.
    # - features: Lista de características especificadas para el modelo (excluye cualquier característica no especificada).
    # - exclude_non_specified_features=True: Usa solo las características en la lista "features".
    # - random_seed=1234: Establece una semilla para garantizar la reproducibilidad de los resultados.
    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=0, # Muy pocos logs
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True, # Usa solo las características en "features"
        random_seed=1234,
    )

    # Entrena el modelo usando el conjunto de datos de entrenamiento.
    model.fit(train_ds)

    # Obtiene el inspector del modelo para evaluar el rendimiento.
    self_evaluation = model.make_inspector().evaluation()


    print(f"Precisión: {self_evaluation.accuracy} Pérdida: {self_evaluation.loss}")


    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)

    return {"message": f"La presisción del modelo es {self_evaluation.accuracy} y la pérdida es {self_evaluation.loss}"}

@app.post("/predict")
async def predict(model_params: Annotated[ModelParams, Body(embed=True)]):
  
    try:
        # Prepare serving DataFrame
        serving_df = {
            'Pclass': [model_params.pclass],
            'Name': [model_params.name],
            'Sex': [model_params.sex],
            'Age': [model_params.age],
            'SibSp': [model_params.sibsp],
            'Parch': [model_params.parch],
            'Fare': [model_params.fare],
            'Cabin': [model_params.cabin],
            'Embarked': [model_params.embarked],
            'Ticket': [model_params.ticket]
        }
        serving_df = pd.DataFrame.from_dict(serving_df)
        print("Input DataFrame:\n", serving_df)

        # Preprocessing
        preprocessed_serving_df = preprocess(serving_df)
        serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)
        
        # Load the model
        with open("model.pkl", 'rb') as file:  # Ensure 'model_pkl_file' is defined as "model.pkl" or a valid path
            model = pickle.load(file)

        # Predictions
        proba_survive = model.predict(serving_ds, verbose=0)[:, 0]
        print("Survival Probabilities:", proba_survive)
        if  proba_survive[0] == 0:
            return {"msg": f"You're dead, your survival probability is {proba_survive[0]}", "survival_probability": proba_survive.tolist()}  # Return JSON serializable response
        else:
            return {"survival_probability": proba_survive.tolist()}

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
    
