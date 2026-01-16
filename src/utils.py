#Utils we'll have all the common things

import os
import sys

import numpy as np 


import pandas as pd
import pickle # this will help us create the pkl file
# pickle: It converts Python objects
# (like trained models, scalers, encoders) into a byte stream that can be saved to disk,
# then loaded back later. Think of it as "freezing" your object to use it later.
# Pickle lets you separate training time (expensive, one-time)
# from inference time (cheap, repeated). Without it, you couldn't build practical ML applications.

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import roc_auc_score, f1_score

def save_object(file_path, obj):
    """
    file_path: A string like "artifacts/model.pkl" - where to save the file

    obj: The Python object you want to save (could be a model, preprocessor, dictionary, etc.)
    """
    try:
        dir_path = os.path.dirname(file_path)
#         os.path.dirname(): Extracts the directory path from the full file path.
# Example: If file_path = "artifacts/models/model.pkl", then dir_path = "artifacts/models"
# This is needed because we need to ensure the directory exists before saving the file.

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
# "with" = Automatically manage opening/closing
# "open(...)" = Open the file
# "model.pkl" = File path
# "wb" = Write mode + Binary mode
# "as file_obj" = Call it 'file_obj' so I can use it
# pickle.dump(...) = Save the model to that file
# (exit block) = File automatically closes

    except Exception as e: #Catches any exception that occurred in the try block and assigns it to variable e.
        raise CustomException(e, sys)

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def zero_to_nan(X):
    # X arrives as a numpy array (via ColumnTransformer), we convert it properly
    X = np.array(X, dtype=float, copy=True)
    X[X == 0] = np.nan
    return X


"""
Utility functions for ML pipeline
Includes save/load objects, model evaluation, and SHAP explainability
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object as a pickle file.
    
    Parameters
    ----------
    file_path : str
        Path where the object will be saved
    obj : object
        Python object to serialize
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved successfully to: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a pickled object from file.
    
    Parameters
    ----------
    file_path : str
        Path to the pickle file
        
    Returns
    -------
    object
        The deserialized Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Train and evaluate multiple classification models.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    models : dict
        Dictionary of {model_name: model_object}
        
    Returns
    -------
    tuple
        (model_results, trained_models)
        - model_results: dict of {model_name: roc_auc_score}
        - trained_models: dict of {model_name: fitted_model}
    """
    try:
        model_results = {}
        trained_models = {}
        
        logging.info("EVALUATING MODELS...")
        
        for model_name, model in models.items():
            logging.info(f"→ Training {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            # Evaluate with ROC-AUC
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_proba)
                model_results[model_name] = roc_auc
                logging.info(f"ROC-AUC: {roc_auc:.4f}")
            else:
                model_results[model_name] = np.nan
                logging.info(f"No predict_proba available")
        
        return model_results, trained_models
        
    except Exception as e:
        raise CustomException(e, sys)


def export_shap_analysis(
    model,
    X_train_transformed,
    X_test_transformed,
    feature_names,
    n_background=1000,
    n_explain=2000,
    output_dir="artifacts/shap"
):
    """
    Compute and export SHAP values + plots for model explainability.
    
    Parameters
    ----------
    model : trained model
        The model to explain (e.g., LogisticRegression)
    X_train_transformed : np.ndarray or sparse matrix
        Transformed training features
    X_test_transformed : np.ndarray or sparse matrix
        Transformed test features
    feature_names : list
        Feature names after preprocessing
    n_background : int, default=1000
        Number of background samples for SHAP
    n_explain : int, default=2000
        Number of test samples to explain
    output_dir : str, default="artifacts/shap"
        Directory to save SHAP artifacts
        
    Returns
    -------
    dict
        Dictionary with keys: shap_values, global_importance, output_dir
    """
    try:
        logging.info("SHAP ANALYSIS STARTED")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ========== 1. CONVERT TO DATAFRAME ==========
        logging.info("Converting arrays to DataFrames...")
        
        if hasattr(X_train_transformed, "toarray"):
            X_train_df = pd.DataFrame(
                X_train_transformed.toarray(),
                columns=feature_names
            )
            X_test_df = pd.DataFrame(
                X_test_transformed.toarray(),
                columns=feature_names
            )
        else:
            X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        # ========== 2. SAMPLE ==========
        logging.info(f"Sampling {n_background} background + {n_explain} explain points...")
        
        bg = X_train_df.sample(n=min(n_background, len(X_train_df)), random_state=42)
        X_explain = X_test_df.sample(n=min(n_explain, len(X_test_df)), random_state=42)
        
        # ========== 3. COMPUTE SHAP VALUES ==========
        logging.info("Creating SHAP explainer...")
        explainer = shap.Explainer(model, bg)
        
        logging.info(f"Computing SHAP values for {len(X_explain)} samples...")
        shap_values = explainer(X_explain)
        logging.info(f"SHAP values shape: {shap_values.values.shape}")
        
        # ========== 4. SAVE SHAP PICKLE ==========
        shap_pkl_path = os.path.join(output_dir, "shap_values.pkl")
        with open(shap_pkl_path, 'wb') as f:
            pickle.dump(shap_values, f)
        logging.info(f"SHAP pickle saved: {shap_pkl_path}")
        
        # ========== 5. SAVE NUMPY ARRAYS ==========
        np.save(os.path.join(output_dir, "shap_values_array.npy"), shap_values.values)
        np.save(os.path.join(output_dir, "shap_base_values.npy"), shap_values.base_values)
        np.save(os.path.join(output_dir, "shap_data.npy"), shap_values.data)
        logging.info("SHAP arrays saved (.npy)")
        
        # ========== 6. GLOBAL IMPORTANCE CSV ==========
        logging.info("Computing global feature importance...")
        
        global_importance = pd.DataFrame({
            "Feature": feature_names,
            "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0)
        }).sort_values("Mean_Abs_SHAP", ascending=False)
        
        importance_csv_path = os.path.join(output_dir, "global_feature_importance.csv")
        global_importance.to_csv(importance_csv_path, index=False)
        logging.info(f"Global importance saved: {importance_csv_path}")
        logging.info(f"\nTop 10:\n{global_importance.head(10).to_string(index=False)}")
        
        # ========== 7. PLOT: BAR ==========
        logging.info("Generating SHAP bar plot...")
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=20, show=False)
        bar_plot_path = os.path.join(output_dir, "shap_bar_plot.png")
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Bar plot: {bar_plot_path}")
        
        # ========== 8. PLOT: BEESWARM ==========
        logging.info("Generating SHAP beeswarm plot...")
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        beeswarm_plot_path = os.path.join(output_dir, "shap_beeswarm_plot.png")
        plt.tight_layout()
        plt.savefig(beeswarm_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Beeswarm plot: {beeswarm_plot_path}")
        
        # ========== 9. PLOT: WATERFALL (3 EXAMPLES) ==========
        logging.info("Generating waterfall plots...")
        for i in range(min(3, len(shap_values))):
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[i], max_display=15, show=False)
            waterfall_path = os.path.join(output_dir, f"shap_waterfall_passenger_{i}.png")
            plt.tight_layout()
            plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
            plt.close()
        logging.info(f"Waterfall plots: {output_dir}/shap_waterfall_passenger_*.png")
        
        # ========== 10. SANITY CHECK ==========
        logging.info("Running SHAP reconstruction sanity check...")
        
        X_check = X_explain.head(5)
        proba_model = model.predict_proba(X_check)[:, 1]
        
        sv_check = explainer(X_check)
        raw_additive = sv_check.base_values + sv_check.values.sum(axis=1)
        
        # Convert log-odds to probability (for LogisticRegression)
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        proba_shap = sigmoid(raw_additive)
        
        check_df = pd.DataFrame({
            "proba_model": proba_model,
            "proba_shap": proba_shap,
            "abs_diff": np.abs(proba_model - proba_shap)
        })
        
        logging.info(f"\nSHAP Check:\n{check_df.to_string(index=False)}")
        
        if check_df["abs_diff"].max() < 1e-3:
            logging.info("SHAP reconstruction accurate (max diff < 0.001)")
        else:
            logging.warning("SHAP reconstruction has larger diff - check link function")
        
        # ========== SUMMARY ==========
        logging.info("SHAP ANALYSIS COMPLETED")
        logging.info(f"Artifacts saved in: {output_dir}/")
        
        return {
            "shap_values": shap_values,
            "global_importance": global_importance,
            "output_dir": output_dir
        }
        
    except Exception as e:
        logging.exception("Error during SHAP analysis")
        raise CustomException(e, sys)
