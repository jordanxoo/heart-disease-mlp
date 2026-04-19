import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import os

def _save_or_show(path):
    if path:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        plt.savefig(path,dpi=300,bbox_inches="tight")
    else:
        plt.show()


def plot_correlation_heatmap(df,path = None):
    plt.figure(figsize=(12,8))
    corr = df.corr()
    sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0)
    plt.title("Macierz korealcji cech")
    plt.tight_layout()
    _save_or_show(path)

def plot_class_distribution(df,path = None):
    plt.figure(figsize=(6,4))
    df["target"].value_counts().plot(kind="bar",color=["steelblue","tomato"])
    plt.xticks([0,1],["Zdrowy","Chory"],rotation = 0)
    plt.title("Rozklad Klas")
    plt.ylabel("Liczba probek")
    plt.tight_layout()
    _save_or_show(path)

def plot_feature_distributions(df,path=None):
    features = df.drop(columns = ["target"]).columns
    fig,axes = plt.subplots(4,4,figsize=(16,12))
    axes = axes.flatten()

    for i,col in enumerate(features):
        for cls,color,label in [(0,"steelblue","Zdrowy"),(1,"Tomato","Chory")]:
            axes[i].hist(df[df["target"] == cls][col].dropna(),
                         alpha = 0.5, color=color,label=label,bins=20)
        axes[i].set_title(col)
        axes[i].legend()
    for j in range(i+1,len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Rozklad cech wedlug klasy", y = 1.02)
    plt.tight_layout()
    _save_or_show(path)


def plot_scatter_age_thalach(df,path=None):
    plt.figure(figsize=(8,6))
    for cls,color,label in [(0,"steelblue","Zdrowy"),(1,"Tomato","Chory")]:
        subset = df[df["target"] == cls]
        plt.scatter(subset["age"],subset["thalach"],
                    alpha=0.5,color=color,label=label)
        
    plt.xlabel("Wiek")
    plt.ylabel("Maksymalne tetno(tchalach)")
    plt.legend()
    plt.tight_layout()
    _save_or_show(path)

def plot_confusion_matrix(model,X_test,y_test,title="Confusion Matrix", path = None):
    cm = confusion_matrix(y_test,model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm,display_labels=["Zdrowy","Chory"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(path)

def plot_roc_curve(models: dict, X_test,y_test,path=None):
    plt.figure(figsize=(8,6))

    for name,model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr,tpr,_ = roc_curve(y_test,y_prob)
        auc_score = auc(fpr,tpr)
        plt.plot(fpr,tpr,label = f"{name} (AUC = {auc_score:.2f})")

    plt.plot([0,1],[0,1],"k--",label = "Losowy (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Krzywa ROC")
    plt.legend()
    plt.tight_layout()
    _save_or_show(path)


def plot_architecture_comparison(results: list[dict], path= None):
    df = pd.DataFrame(results)[["model","recall","auc_roc"]]
    df = df.set_index("model")

    df.plot(kind = "bar",figsize=(10,6),color=["tomato","steelblue"])
    plt.title("Porownaine architektur: Recall i AUC-ROC")
    plt.ylabel("Wartosci metryki")
    plt.xticks(rotation=30,ha="right")
    plt.ylim(0,1)
    plt.legend(["Recall","AUC-ROC"])
    plt.tight_layout()
    _save_or_show(path)

