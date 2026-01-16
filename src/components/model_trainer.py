"""
Model Trainer Component (Simplified)
Orchestrates training, uses utils for evaluation and explainability
"""

import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Metrics
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Configuration for model training artifacts."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_file_path: str = os.path.join("artifacts", "model_report.csv")


class ModelTrainer:
    """
    Trains multiple classification models, selects best, and optionally exports SHAP.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, threshold=0.6):
        """
        Main training pipeline.
        
        Parameters
        ----------
        train_array : np.ndarray
            Training data (features + target in last column)
        test_array : np.ndarray
            Test data (features + target in last column)
        threshold : float, default=0.6
            Decision threshold for classification
            
        Returns
        -------
        dict
            Training results with best model, metrics, and confusion matrix
        """
        try:
            logging.info("MODEL TRAINING STARTED")

            # ========== 1. SPLIT ARRAYS ==========
            logging.info("Splitting arrays into X and y...")
            
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)
            
            logging.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
            logging.info(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

            # ========== 2. DEFINE MODELS ==========
            logging.info("Defining models...")
            
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=3000,
                    class_weight='balanced',
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "HistGradient Boosting": HistGradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(
                    random_state=42,
                    eval_metric='logloss',
                    n_jobs=-1
                ),
                "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42)
            }
            
            logging.info(f"{len(models)} models defined.")

            # ========== 3. TRAIN & EVALUATE (via utils) ==========
            model_results, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            # ========== 4. SELECT BEST MODEL ==========
            logging.info("SELECTING BEST MODEL...")

            # Create comparison DataFrame
            results_df = pd.DataFrame({
                "Model Name": list(model_results.keys()),
                "ROC-AUC Score": list(model_results.values())
            }).sort_values(by="ROC-AUC Score", ascending=False).reset_index(drop=True)
            
            logging.info("\n" + results_df.to_string(index=False))

            # **PRIORITY: Logistic Regression if ROC-AUC >= 0.6**
            logreg_score = model_results.get("Logistic Regression", 0.0)

            if logreg_score >= 0.6:
                best_model_name = "Logistic Regression"
                best_model_score = logreg_score
                logging.info("→ Logistic Regression selected (interpretability priority)")
            else:
                best_model_name = results_df.iloc[0]["Model Name"]
                best_model_score = results_df.iloc[0]["ROC-AUC Score"]
                logging.info(f"→ LogReg below 0.6, selecting best: {best_model_name}")

            best_model = trained_models[best_model_name]

            logging.info(f"BEST MODEL: {best_model_name}")
            logging.info(f"   ROC-AUC: {best_model_score:.4f}")

            # Check minimum threshold
            if best_model_score < 0.60:
                raise CustomException(
                    f"No model meets threshold! ROC-AUC={best_model_score:.4f} < 0.60",
                    sys
                )

            # ========== 5. DETAILED METRICS ==========
            logging.info("Computing detailed metrics...")

            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= threshold).astype(int)

            roc_auc_final = roc_auc_score(y_test, y_test_proba)
            f1_dissatisfied = f1_score(y_test, y_test_pred, pos_label=0)
            f1_satisfied = f1_score(y_test, y_test_pred, pos_label=1)
            f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

            logging.info(f"METRICS - {best_model_name}")
            logging.info(f"ROC-AUC:            {roc_auc_final:.4f}")
            logging.info(f"F1 (Dissatisfied):  {f1_dissatisfied:.4f}  ← Service Recovery")
            logging.info(f"F1 (Satisfied):     {f1_satisfied:.4f}")
            logging.info(f"F1 (Weighted):      {f1_weighted:.4f}")
            logging.info(f"Threshold:          {threshold}")

            # Classification report
            report = classification_report(
                y_test, y_test_pred,
                target_names=["Dissatisfied (0)", "Satisfied (1)"]
            )
            logging.info("\n" + report)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            logging.info("\nCONFUSION MATRIX:")
            logging.info(f"  TN: {tn:>6,}  FP: {fp:>6,}")
            logging.info(f"  FN: {fn:>6,}  TP: {tp:>6,}")

            # ========== 6. SAVE MODEL ==========
            logging.info("Saving best model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # ========== 7. SAVE REPORT ==========
            results_df.to_csv(
                self.model_trainer_config.model_report_file_path,
                index=False
            )
            logging.info(f"Model report saved: {self.model_trainer_config.model_report_file_path}")

            logging.info("MODEL TRAINING COMPLETED")

            # ========== 8. RETURN RESULTS ==========
            return {
                "best_model_name": best_model_name,
                "best_model": best_model,
                "roc_auc": roc_auc_final,
                "f1_dissatisfied": f1_dissatisfied,
                "f1_satisfied": f1_satisfied,
                "f1_weighted": f1_weighted,
                "threshold": threshold,
                "confusion_matrix": cm,
                "X_train": X_train,  # For SHAP export
                "X_test": X_test
            }

        except Exception as e:
            logging.exception("Error during model training")
            raise CustomException(e, sys)

