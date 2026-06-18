import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

def train_and_save_models(data_path, models_dir):
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    
    # Define features
    categorical_features = ['event_type', 'event_cause', 'corridor', 'police_station']
    numerical_features = [
        'latitude', 'longitude', 'hour_of_day', 'day_of_week', 'is_weekend', 
        'month', 'junction_event_frequency', 'police_station_event_frequency', 
        'has_end_coords', 'description_length', 'planned_lead_time_hours'
    ]
    
    features = categorical_features + numerical_features
    
    # Make sure we don't have NaN in features
    # Impute missing categoricals with 'Unknown' and numericals with median
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
        
    os.makedirs(models_dir, exist_ok=True)
    
    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    # -------------------------------------------------------------
    # MODEL 1: Priority Classifier (High vs Low)
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Priority Classifier ---")
    X = df[features]
    y_priority = df['priority'].apply(lambda x: 1 if str(x).lower() == 'high' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
    )
    
    priority_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    priority_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = priority_pipeline.predict(X_test)
    print("Priority Classifier Evaluation:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
    
    # Save model
    priority_model_path = os.path.join(models_dir, 'priority_model.joblib')
    joblib.dump(priority_pipeline, priority_model_path)
    print(f"Priority model saved to {priority_model_path}")
    
    # -------------------------------------------------------------
    # MODEL 2: Road Closure Predictor
    # -------------------------------------------------------------
    print("\n--- Training Model 2: Road Closure Predictor ---")
    y_closure = df['requires_road_closure']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_closure, test_size=0.2, random_state=42, stratify=y_closure
    )
    
    closure_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    closure_pipeline.fit(X_train_c, y_train_c)
    
    # Evaluate
    y_pred_c = closure_pipeline.predict(X_test_c)
    print("Road Closure Predictor Evaluation:")
    print(classification_report(y_test_c, y_pred_c, target_names=['No Closure', 'Closure Required']))
    
    # Save model
    closure_model_path = os.path.join(models_dir, 'closure_model.joblib')
    joblib.dump(closure_pipeline, closure_model_path)
    print(f"Road closure model saved to {closure_model_path}")
    
    # -------------------------------------------------------------
    # MODEL 3: Resolution Duration Regressor
    # -------------------------------------------------------------
    print("\n--- Training Model 3: Resolution Duration Regressor ---")
    # Filter for rows that have valid resolution duration
    duration_df = df[df['resolution_duration_minutes'].notnull()].copy()
    print(f"Data points with valid duration: {len(duration_df)}")
    
    if len(duration_df) > 50:
        X_dur = duration_df[features]
        y_dur = duration_df['resolution_duration_minutes']
        
        # Log-transform target to handle skewness
        y_dur_log = np.log1p(y_dur)
        
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_dur, y_dur_log, test_size=0.2, random_state=42
        )
        
        duration_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        duration_pipeline.fit(X_train_d, y_train_d)
        
        # Evaluate
        y_pred_d = duration_pipeline.predict(X_test_d)
        # Convert back from log scale for evaluation metrics
        y_test_d_orig = np.expm1(y_test_d)
        y_pred_d_orig = np.expm1(y_pred_d)
        
        print("Duration Regressor Evaluation (in minutes):")
        print(f"MAE: {mean_absolute_error(y_test_d_orig, y_pred_d_orig):.2f} mins")
        print(f"Median Abs Error: {np.median(np.abs(y_test_d_orig - y_pred_d_orig)):.2f} mins")
        print(f"R2 Score: {r2_score(y_test_d, y_pred_d):.4f} (log-scale)")
        
        # Save model
        duration_model_path = os.path.join(models_dir, 'duration_model.joblib')
        joblib.dump(duration_pipeline, duration_model_path)
        print(f"Duration model saved to {duration_model_path}")
    else:
        print("Not enough data to train duration regressor. Saving dummy/mean model.")
        
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    data_path = "data/processed_data.csv"
    models_dir = "models"
    train_and_save_models(data_path, models_dir)
