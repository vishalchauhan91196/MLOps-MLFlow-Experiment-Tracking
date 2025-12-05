import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='vishalchauhan91196', repo_name='MLOps-MLFlow-Experiment-Tracking', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/vishalchauhan91196/MLOps-MLFlow-Experiment-Tracking.mlflow")
mlflow.set_experiment('Mlflow-exp-tracking')

mlflow.sklearn.autolog()

# Load Wine dataset
wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=45)

# Define params for RF Model
max_depth = 8
n_estimators = 10

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=45)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot
    plt.savefig('Confusion-matrix.png')

    # Log script file using mlflow
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({'Author': 'Vishal', 'Project': 'Wine Classification'})

    print(accuracy)