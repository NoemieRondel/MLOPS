import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression

# Création d'un modèle de test
model = LinearRegression()
model.fit([[0, 0], [1, 1]], [0, 1])

# Démarrer une session MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "test_model")
    mlflow.log_param("test_param", "test_value")
    print("Modèle de test et paramètre enregistrés.")
