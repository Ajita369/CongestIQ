import pandas as pd
import numpy as np
import os
import joblib
import json
from pure_tree import PureDecisionTree, PureRandomForest

def train_and_save_pure_models(data_path, models_dir):
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    
    categorical_features = ['event_type', 'event_cause', 'corridor', 'police_station']
    numerical_features = [
        'latitude', 'longitude', 'hour_of_day', 'day_of_week', 'is_weekend', 
        'month', 'junction_event_frequency', 'police_station_event_frequency', 
        'has_end_coords', 'description_length', 'planned_lead_time_hours'
    ]
    
    # 1. Build and save categorical mappings (Label Encoding)
    print("Mapping categorical features...")
    mappings = {}
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
        unique_vals = sorted(df[col].unique())
        # Make a map mapping value -> integer index
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        # Add default for unknown categories at inference
        mapping['Unknown'] = mapping.get('Unknown', len(unique_vals))
        mappings[col] = mapping
        df[col] = df[col].map(mapping)
        
    # Save mappings
    os.makedirs(models_dir, exist_ok=True)
    mappings_path = os.path.join(models_dir, 'mappings.joblib')
    joblib.dump(mappings, mappings_path)
    print(f"Categorical mappings saved to {mappings_path}")
    
    # 2. Impute numerical columns
    print("Imputing numerical features...")
    imputations = {}
    for col in numerical_features:
        median_val = df[col].median()
        if pd.isnull(median_val):
            median_val = 0.0
        imputations[col] = median_val
        df[col] = df[col].fillna(median_val)
        
    # Save imputations
    imputations_path = os.path.join(models_dir, 'imputations.joblib')
    joblib.dump(imputations, imputations_path)
    
    # Prepare features matrix (numpy arrays)
    features = categorical_features + numerical_features
    X = df[features].values
    
    # Train/Test Split (Custom simple split to avoid sklearn train_test_split DLL issue if any,
    # though train_test_split is pure Python, we can write a simple numpy train/test split for safety)
    print("Splitting data into train/test sets...")
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split = int(0.8 * len(df))
    train_idx, test_idx = indices[:split], indices[split:]
    
    # -------------------------------------------------------------
    # MODEL 1: Priority Classifier (High vs Low)
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Priority Classifier (Pure Tree) ---")
    y_priority = df['priority'].apply(lambda x: 1 if str(x).lower() == 'high' else 0).values
    
    X_train, y_train = X[train_idx], y_priority[train_idx]
    X_test, y_test = X[test_idx], y_priority[test_idx]
    
    # Using DecisionTree to keep training fast, depth 7
    priority_model = PureDecisionTree(max_depth=7, min_samples_split=5, mode='classifier')
    priority_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = priority_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Priority Classifier Accuracy: {accuracy * 100:.2f}%")
    # Basic confusion matrix
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    priority_model_path = os.path.join(models_dir, 'priority_model.joblib')
    joblib.dump(priority_model, priority_model_path)
    print(f"Priority model saved to {priority_model_path}")
    
    # -------------------------------------------------------------
    # MODEL 2: Road Closure Predictor
    # -------------------------------------------------------------
    print("\n--- Training Model 2: Road Closure Predictor (Pure Tree) ---")
    y_closure = df['requires_road_closure'].values
    
    y_train_c = y_closure[train_idx]
    y_test_c = y_closure[test_idx]
    
    closure_model = PureDecisionTree(max_depth=7, min_samples_split=5, mode='classifier')
    closure_model.fit(X_train, y_train_c)
    
    # Evaluate
    y_pred_c = closure_model.predict(X_test)
    accuracy_c = np.mean(y_pred_c == y_test_c)
    print(f"Road Closure Predictor Accuracy: {accuracy_c * 100:.2f}%")
    tp_c = np.sum((y_test_c == 1) & (y_pred_c == 1))
    fp_c = np.sum((y_test_c == 0) & (y_pred_c == 1))
    fn_c = np.sum((y_test_c == 1) & (y_pred_c == 0))
    tn_c = np.sum((y_test_c == 0) & (y_pred_c == 0))
    print(f"Confusion Matrix: TP={tp_c}, FP={fp_c}, FN={fn_c}, TN={tn_c}")
    
    # Save model
    closure_model_path = os.path.join(models_dir, 'closure_model.joblib')
    joblib.dump(closure_model, closure_model_path)
    print(f"Road closure model saved to {closure_model_path}")
    
    # -------------------------------------------------------------
    # MODEL 3: Resolution Duration Regressor
    # -------------------------------------------------------------
    print("\n--- Training Model 3: Resolution Duration Regressor (Pure Tree) ---")
    duration_df = df[df['resolution_duration_minutes'].notnull()].copy()
    print(f"Data points with valid duration: {len(duration_df)}")
    
    if len(duration_df) > 50:
        X_dur = duration_df[features].values
        y_dur = duration_df['resolution_duration_minutes'].values
        y_dur_log = np.log1p(y_dur)
        
        # Split
        indices_d = np.arange(len(duration_df))
        np.random.shuffle(indices_d)
        split_d = int(0.8 * len(duration_df))
        train_idx_d, test_idx_d = indices_d[:split_d], indices_d[split_d:]
        
        X_train_d, y_train_d = X_dur[train_idx_d], y_dur_log[train_idx_d]
        X_test_d, y_test_d = X_dur[test_idx_d], y_dur_log[test_idx_d]
        
        duration_model = PureDecisionTree(max_depth=7, min_samples_split=5, mode='regressor')
        duration_model.fit(X_train_d, y_train_d)
        
        # Evaluate
        y_pred_d = duration_model.predict(X_test_d)
        y_test_d_orig = np.expm1(y_test_d)
        y_pred_d_orig = np.expm1(y_pred_d)
        
        mae = np.mean(np.abs(y_test_d_orig - y_pred_d_orig))
        median_ae = np.median(np.abs(y_test_d_orig - y_pred_d_orig))
        print(f"Duration Regressor MAE: {mae:.2f} mins")
        print(f"Duration Regressor Median Absolute Error: {median_ae:.2f} mins")
        
        # Save model
        duration_model_path = os.path.join(models_dir, 'duration_model.joblib')
        joblib.dump(duration_model, duration_model_path)
        print(f"Duration model saved to {duration_model_path}")
    else:
        print("Not enough data to train duration regressor.")
        
    print("\nAll custom models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_pure_models("data/processed_data.csv", "models")
