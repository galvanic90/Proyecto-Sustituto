import requests

endpoint = 'http://127.0.0.1:8000'
response = requests.get(endpoint)

assert response.json()["message"] == "Probability of surviving the titanic ML API"

# Consumo del endpoint para el entrenamiento del modelo 
train_endpoint = endpoint + "/train"
response_t = requests.get(train_endpoint)
print("Model trained params:", response_t.json()["message"])

# Consumo del endpoint que permite realizar las predicciones
pred_endpoint = endpoint + "/predict"
payload = {
    "model_params": {
    "pclass": 3,
    "name": "Kelly, Mr. James",
    "sex": "male",
    "age": 34.5,
    "sibsp": 0,
    "parch": 0,
    "fare": 7.8292,
    "cabin": "",
    "embarked": "Q",
    "ticket": "315154"
}}
response_predict = requests.post(pred_endpoint, json=payload)
print("prediction 1", response_predict.json())

payload_2 = {
    "model_params": {
    "pclass": 1,
    "name": "Cumings, Mr. John Bradley",
    "sex": "male",
    "age": 39,
    "sibsp": 1,
    "parch": 0,
    "fare": 71.2833,
    "cabin": "C85",
    "embarked": "C",
    "ticket": "PC 17599"
}}
response_predict_2 = requests.post(pred_endpoint, json=payload)
print("prediction 2", response_predict_2.json())
