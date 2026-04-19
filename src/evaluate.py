from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
import os
import json
def evaluate_model(model,X_test,y_test,model_name: str) -> dict:

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    auc_roc = roc_auc_score(y_test,y_prob[:,1])
    cm = confusion_matrix(y_test,y_pred)

    return {
        "model" : model_name,
        "accuracy" : accuracy,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "auc_roc"  : auc_roc,
        "confusion_matrix" : cm.tolist()
    }


def save_metrics(results: list[dict], path: str = "/results,metrics.json"):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open (path,"w") as f:
        json.dump(results,f,indent=2)