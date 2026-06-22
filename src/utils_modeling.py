import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, precision_recall_curve, roc_curve, precision_score, recall_score, brier_score_loss, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


# ---- Function to perform Stratified Cross-validation ----

def stratified_CV(model, X_train, y_train, n_splits):
    
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    pr_aucs = []
    roc_aucs = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_v)[:,1]
        
        pr_auc = average_precision_score(y_v, probs)
        roc_auc = roc_auc_score(y_v, probs)

        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)

    pr_auc = np.mean(pr_aucs)
    roc_auc = np.mean(roc_aucs)

    return pr_auc, roc_auc


# ---- Function to perform Hyperparamater tuning ----

def hyperparameter_tune(model, param_grid, X_train, y_train, iters):

    # Random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=iters,                       # n of random combinations
        scoring='average_precision',        # PR-AUC
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=1
    )

    # Train random search
    random_search.fit(X_train, y_train)

    # Best results
    best_score = random_search.best_score_
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_score, best_params, best_model


# ---- Function to perform Threshold tuning to obtain the desired recall ----

def select_threshold_for_recall(y_true, probs, target_recall):

    # Compute precision-recall curve 
    prec, rec, th = precision_recall_curve(y_true, probs)
    recall = rec[:-1]
    valid_idx = np.where(recall >= target_recall)[0]

    if len(valid_idx) == 0:
        return 0.5

    return th[valid_idx[-1]]


# ---- Function to calculate Evaluatuion metrics ----

def evaluate_model(y_test, y_probs, y_pred, name):

    # Classification report and Confusion matrix
    print('\nClassification report:\n', classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    # ROC-CURVE and PR-CURVE
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    # Scores
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_probs)
    flag_rate = y_pred.mean()

    return roc_auc, pr_auc, precision, recall, f1, brier, flag_rate


# ---- Function to plot ROC-AUC and PR-AUC curves ----

def plot_pr_roc_auc(y_test, y_probs, name):

    # ROC-CURVE and PR-CURVE
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test,y_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    axes[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # PRECISION-RECALL curve
    axes[1].plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'{name} Evaluation',  size=16)
    plt.tight_layout()
    plt.show()


# ---- Function to papply Calibration to models ----
def calibrate_model(model, X_train, y_train, method):
    calibrated_model = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


# ---- Function to plot the Calibration curve ----
def calibration_curve_plot(probs_test, y_test, name):

    prob_true, prob_pred = calibration_curve(y_test, probs_test, n_bins=10)

    plt.figure(figsize=(6,4))

    plt.plot(prob_pred, prob_true, marker='o', label=f'{name}')              # Calibration curve
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect calibration')   # Perfect calibration line

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed mortality")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()