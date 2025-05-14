
# Modelo Predictivo - Diagnóstico Médico Final

Este proyecto implementa un modelo de red neuronal en TensorFlow/Keras para predecir el diagnóstico médico final de conductores, usando datos de salud. También se proporciona una interfaz gráfica utilizando **Gradio**.

---

## 📦 Librerías Utilizadas

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
```

---

## 📁 Paso 1: Cargar los datos

Se carga el archivo CSV `conductores_salud.csv` y se convierten las variables categóricas a numéricas.

```python
def cargar_datos():
    datos = pd.read_csv("conductores_salud.csv")
    datos['Género'] = datos['Género'].map({'Masculino': 1, 'Femenino': 0})
    datos['Consumo de Tabaco'] = datos['Consumo de Tabaco'].map({'Sí': 1, 'No': 0})
    X = datos.drop('Diagnóstico Médico Final', axis=1)
    y = datos['Diagnóstico Médico Final']
    return X, y
```

---

## ⚙️ Paso 2: Preprocesamiento

Se escalan los datos y se dividen en conjunto de entrenamiento y prueba.

```python
def preprocesar_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
```

---

## 🧠 Paso 3: Crear Modelo de Red Neuronal

Se crea un modelo secuencial con capas densas y dropout.

```python
def crear_modelo(input_shape):
    modelo = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo
```

---

## 📈 Paso 4: Entrenar el Modelo

```python
def entrenar_modelo(modelo, X_train, y_train):
    historia = modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    return historia
```

---

## ✅ Paso 5: Evaluar el Modelo

```python
def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    predicciones_binarias = (predicciones > 0.5).astype(int)
    print(classification_report(y_test, predicciones_binarias))
    sns.heatmap(confusion_matrix(y_test, predicciones_binarias), annot=True)
    plt.show()
```

---

## 📊 Gráfico de Precisión

```python
def graficar_precision(historia):
    plt.plot(historia.history['accuracy'], label='Entrenamiento')
    plt.plot(historia.history['val_accuracy'], label='Validación')
    plt.legend()
    plt.show()
```

---

## 🧪 Interfaz con Gradio

```python
def predecir_diagnostico(...):
    entrada = [[...]]
    entrada_scaled = scaler.transform(entrada)
    probabilidad = modelo.predict(entrada_scaled)[0][0]
    return {"Alto Riesgo": float(probabilidad), "Bajo Riesgo": float(1 - probabilidad)}
```

---

## 🚀 Función Principal

```python
def main():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = preprocesar_datos(X, y)
    modelo = crear_modelo(X_train.shape[1])
    historia = entrenar_modelo(modelo, X_train, y_train)
    evaluar_modelo(modelo, X_test, y_test)
    graficar_precision(historia)
    interfaz = gr.Interface(...)
    interfaz.launch()
```

---

## ▶️ Ejecución

```python
if __name__ == '__main__':
    main()
```
