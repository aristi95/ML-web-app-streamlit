# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("🔧 Entrenando modelo de Iris...")

# Cargar datos
iris = load_iris()
X = iris.data
y = iris.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Guardar modelo
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print(f"✅ Modelo entrenado y guardado!")
print(f"📊 Precisión: {accuracy:.4f}")
print(f"🎯 Clases: {iris.target_names}")
print("📁 Archivo guardado: iris_model.pkl")