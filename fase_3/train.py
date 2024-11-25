# Importa la librería NumPy para realizar operaciones matemáticas y manipulación de matrices
import numpy as np

# Importa la librería pandas para el manejo y análisis de estructuras de datos (DataFrames)
import pandas as pd

# Importa la librería os para interactuar con el sistema operativo, como acceder a directorios y archivos
import os

# Importa la librería TensorFlow, que es utilizada para construir y entrenar modelos de Machine Learning
import tensorflow as tf

# Importa TensorFlow Decision Forests, una librería especializada en modelos basados en bosques aleatorios y árboles de decisión
import tensorflow_decision_forests as tfdf
import pickle
import os

project_dir = os.path.join(os.path.dirname(__file__))
train_data_path = os.path.join(project_dir, "train.csv")

# Carga el conjunto de datos de entrenamiento del archivo CSV ubicado en la carpeta de entrada.
# train.csv contiene los datos históricos de los pasajeros del Titanic (características y si sobrevivieron o no).
train_df = pd.read_csv(train_data_path)

# Carga el conjunto de datos de prueba (serving_df), que no incluye la columna de "Supervivencia" (Survived).
# test.csv se utiliza para hacer predicciones basadas en las características de los pasajeros.

# Define una función para preprocesar el conjunto de datos.
def preprocess(df):
    # Crea una copia del DataFrame para evitar modificar el original directamente.
    df = df.copy()

    # Normaliza los nombres eliminando caracteres especiales como comas, paréntesis, corchetes y comillas.
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

    # Extrae el número de boleto, que se encuentra al final del string en la columna 'Ticket'.
    def ticket_number(x):
        return x.split(" ")[-1]

    # Extrae los componentes del boleto, eliminando el número (último elemento).
    # Si no hay componentes adicionales, se asigna "NONE".
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])

    # Aplica la normalización de nombres en la columna 'Name'.
    df["Name"] = df["Name"].apply(normalize_name)

    # Crea una nueva columna 'Ticket_number' con el número del boleto extraído.
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)

    # Crea una nueva columna 'Ticket_item' con el ítem del boleto (sin el número final).
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)

    # Devuelve el DataFrame preprocesado.
    return df

# Aplica la función de preprocesamiento al conjunto de datos de entrenamiento.
preprocessed_train_df = preprocess(train_df)


# Muestra las primeras 5 filas del conjunto de datos de entrenamiento preprocesado.
preprocessed_train_df.head(5)

# Crea una lista de las columnas (características) del conjunto de datos preprocesado.
input_features = list(preprocessed_train_df.columns)

# Elimina la columna 'Ticket', ya que no se utilizará directamente como característica para el modelo.
input_features.remove("Ticket")

# Elimina la columna 'PassengerId', ya que es un identificador único que no aporta valor predictivo.
input_features.remove("PassengerId")

# Elimina la columna 'Survived', ya que es la etiqueta objetivo (lo que queremos predecir).
input_features.remove("Survived")

# La columna 'Ticket_number' podría ser eliminada si no se desea incluirla, aquí está comentada.
# Se podría descomentar si se quiere excluirla como característica.
# input_features.remove("Ticket_number")

# Imprime las características de entrada seleccionadas que serán utilizadas por el modelo.
# print(f"Características de entrada: {input_features}")

# Define una función para dividir los nombres en tokens. TensorFlow Decision Forests (TF-DF) puede manejar tokens de texto nativamente.
def tokenize_names(features, labels=None):
    """Divide los nombres en tokens. TF-DF puede consumir tokens de texto de manera nativa."""
    features["Name"] = tf.strings.split(features["Name"])  # Divide los nombres en tokens usando espacios como separadores
    return features, labels

# Convierte el DataFrame preprocesado de entrenamiento a un conjunto de datos de TensorFlow.
# Especifica la columna 'Survived' como la etiqueta (variable objetivo) y aplica la función de tokenización.
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df, label="Survived").map(tokenize_names)

# Convierte el DataFrame preprocesado de prueba a un conjunto de datos de TensorFlow.
# No se especifica una etiqueta, ya que este conjunto de datos se utiliza para hacer predicciones.
#serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)

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

# Imprime la precisión y la pérdida del modelo en el conjunto de entrenamiento.
# La precisión mide la exactitud de las predicciones, mientras que la pérdida evalúa el error del modelo.
print(f"Precisión: {self_evaluation.accuracy} Pérdida: {self_evaluation.loss}")

# Define un modelo de Gradient Boosted Trees usando TensorFlow Decision Forests con parámetros adicionales.
# Configura el modelo con los siguientes parámetros:
# - verbose=0: Muestra muy pocos logs durante el entrenamiento.
# - features: Lista de características especificadas para el modelo (usa solo las características en "features").
# - exclude_non_specified_features=True: Usa solo las características en la lista "features".
# - min_examples=1: Número mínimo de ejemplos necesarios para realizar una división en el árbol.
# - categorical_algorithm="RANDOM": Algoritmo utilizado para manejar variables categóricas (aleatorio en este caso).
# - shrinkage=0.05: Tasa de reducción del gradiente (shrinkage) para mejorar el rendimiento del modelo.
# - split_axis="SPARSE_OBLIQUE": Tipo de partición utilizada para dividir las características en árboles oblicuos dispersos.
# - sparse_oblique_normalization="MIN_MAX": Normalización aplicada a las características oblicuas dispersas.
# - sparse_oblique_num_projections_exponent=2.0: Exponente utilizado para el número de proyecciones oblicuas dispersas.
# - num_trees=2000: Número de árboles en el modelo.
# - random_seed=1234: Establece una semilla para garantizar la reproducibilidad de los resultados.

model = tfdf.keras.GradientBoostedTreesModel(
    verbose=0, # Muy pocos logs
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    exclude_non_specified_features=True, # Usa solo las características en "features"
    min_examples=1,
    categorical_algorithm="RANDOM",
    shrinkage=0.05,
    split_axis="SPARSE_OBLIQUE",
    sparse_oblique_normalization="MIN_MAX",
    sparse_oblique_num_projections_exponent=2.0,
    num_trees=2000,
    random_seed=1234,
)

# Entrena el modelo utilizando el conjunto de datos de entrenamiento.
model.fit(train_ds)

# Obtiene el inspector del modelo para evaluar su rendimiento en el conjunto de entrenamiento.
self_evaluation = model.make_inspector().evaluation()

# Imprime la precisión y la pérdida del modelo en el conjunto de entrenamiento.
# La precisión indica la exactitud de las predicciones del modelo, mientras que la pérdida evalúa el error del modelo.
print(f"Precisión: {self_evaluation.accuracy} Pérdida: {self_evaluation.loss}")

model_pkl_file = "model.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(model, file)

