# Proyecto-Sustituto
Este repositorio contiene las entregas correspondientes del proyecto sustituto de las asignaturas que se listan acontinuación

- Asignaturas: Introdución a la inteligencia  - Modelos 1

## 👩🏾‍💻👨‍💻  Integrantes

- Nombre: Sara Galván Ortega 
- Número de documento: 1032440925
- Programa académico: Ingeniería de sistemas
---
- Nombre: Emmanuel Narváez Hernandez 
- Número de documento: 1000548307
- Programa académico: Ingeniería de sistemas


# Competencia Titanic con TensorFlow Decision Forests

## Descripción del general proyecto 
Este proyecto entrena un modelo de Gradient Boosted Trees utilizando TensorFlow Decision Forests para predecir la supervivencia en la competencia Titanic de Kaggle. A continuación, se describen los pasos necesarios para ejecutar el proyecto correctamente.

## 🔥 FASE 1 

### 1. Instalación de Dependencias

Asegúrate de instalar TensorFlow Decision Forests en tu entorno. Utiliza el comando correspondiente para instalar la biblioteca necesaria.

### 2. Carga del Conjunto de Datos

Descarga los datos necesarios para la competencia Titanic desde Kaggle y coloca los archivos CSV en el directorio especificado. Asegúrate de que los archivos `train.csv` y `test.csv` estén disponibles para su uso en el proyecto.

### 3. Preparación del Conjunto de Datos

Realiza las siguientes transformaciones en el conjunto de datos:
1. **Tokenización de Nombres**: Convierte los nombres de los pasajeros en tokens para facilitar el procesamiento.
2. **Extracción de Prefijo del Ticket**: Separa el prefijo y el número del ticket para su análisis.

### 4. Conversión del Conjunto de Datos a TensorFlow Dataset

Convierte los DataFrames de Pandas en conjuntos de datos de TensorFlow. Asegúrate de que el conjunto de datos de entrenamiento incluya las etiquetas y el conjunto de datos de prueba no tenga etiquetas.

### 5. Entrenamiento del Modelo con Parámetros Predeterminados

Entrena un modelo de Gradient Boosted Trees utilizando los parámetros predeterminados proporcionados. Asegúrate de registrar las métricas de evaluación del modelo, como la precisión y la pérdida.

### 6. Entrenamiento del Modelo con Parámetros Mejorados

Ajusta algunos parámetros específicos del modelo para mejorar el rendimiento. Reentrena el modelo utilizando estos parámetros ajustados y evalúa su desempeño.

### 7. Ajuste de Hiperparámetros

Configura y ejecuta un proceso de ajuste de hiperparámetros para encontrar la mejor combinación de parámetros para el modelo. Utiliza un buscador de hiperparámetros para explorar diferentes configuraciones y ajusta el modelo en consecuencia.

### 8. Creación de un Ensamble

Crea un ensamble de modelos entrenando múltiples modelos con diferentes semillas. Combina las predicciones de estos modelos para mejorar la precisión global. Asegúrate de utilizar la técnica de regularización adecuada para el ensamble.

### 9. Realización de Predicciones

Genera las predicciones para el conjunto de datos de prueba utilizando el modelo de ensamble. Convierte estas predicciones en el formato requerido para la competencia y crea un archivo de sumisión.

### 10. Evaluación del Modelo

Verifica el archivo de sumisión para asegurarte de que esté correctamente formateado y contiene las predicciones esperadas. Revisa el contenido del archivo para confirmar que las predicciones están alineadas con las expectativas del desafío.

## 🔥🔥 FASE 2
A continuación se explican los pasos para ejecutar la fase 2 del proyecto sustituto. Dado que para esta fase del proyecto sustituto ya se ha implementado Docker :whale: los pasos para la ejecución son sencillos. 

### 1. Construir la imagen de Docker

Una vez se ha clonado este repositorio en su máquina local, verifique que tiene Docker instalado, ejecutando el siguiente comando en su terminal:

```bash

docker ps

``` 

Este comando se emplea para verificar los procesos que docker está corriendo en su máquina. Una vez se ha hecho esta verificación, ejecute este comando para construír la imagen de docker 

```bash

docker build -t fase_2 .

``` 

### 2. Verificar la recién creada imagen de Docker

Para verificar la recién creada imagen de docker ejecute el siguiente comando en su terminal:

```bash

docker images

```

Este comando sirve para listar todas las imágenes de docker disponibles en su máquina, tal vez inicialmente no le vea utilidad, pero a medida que tenga más proyectos dentro de contenedores, le será cada vez más útil.

### 3. Ejecutar la imagen de :whale: Docker 

Para ejecutar la imagen de docker escriba el siguiente comando en su terminal:

```bash

docker run fase_2 
```

Una vez termine la ejecución se podrá observar en la terminal una pequeña muestra de las predicciones realizadas con el modelo de machine learning. 
