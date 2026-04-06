import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd


def plot_confusion_matrix(y_pred, y_test, model_name="model"):
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(
        cm,
        index=["FALSE", "TRUE"],      # Actual labels: 0 -> F, 1 -> T
        columns=["FALSE", "TRUE"]     # Predicted labels: 0 -> F, 1 -> T
    )
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.show()
    os.makedirs(f"plots/{model_name}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_f1_score_by_class(y_pred, y_test, model_name="model"):
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_scores = {label: metrics['f1-score'] for label, metrics in report.items() if label not in ['accuracy', 'macro avg', 'weighted avg']}
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='viridis')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Class')
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(f"plots/{model_name}", exist_ok=True)
    plt.savefig(f"plots/{model_name}/f1_score_by_class.png", dpi=300, bbox_inches="tight")
    plt.close()
    
def plot(y_pred, y_test, model_name="model"):
    
    # plot confusion matrix
    plot_confusion_matrix(y_pred, y_test, model_name=model_name)
    
    # f1 score by class
    plot_f1_score_by_class(y_pred, y_test, model_name=model_name)
    
    
def feature_importance(feature_importances, feature_names, model_name="model"):
    import matplotlib.pyplot as plt

    os.makedirs(f"plots/{model_name}", exist_ok=True)

    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    print("Feature Importances:")
    for feature, importance in sorted_features.items():
        print(f"{feature}: {importance:.4f}")

    # sorted_features = {key.replace("DATA", "BYTE"): value for key, value in sorted_features.items()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(sorted_features.keys()), y=list(sorted_features.values()), palette="viridis")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


    
    