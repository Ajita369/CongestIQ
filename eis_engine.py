import os
import joblib
import numpy as np

# Import the custom classes so joblib can deserialize them
from pure_tree import PureDecisionTree, PureRandomForest

class EISEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.priority_model = None
        self.closure_model = None
        self.duration_model = None
        self.mappings = None
        self.imputations = None
        self.load_artifacts()

    def load_artifacts(self):
        print("EISEngine: Loading model artifacts...")
        self.priority_model = joblib.load(os.path.join(self.models_dir, 'priority_model.joblib'))
        self.closure_model = joblib.load(os.path.join(self.models_dir, 'closure_model.joblib'))
        self.duration_model = joblib.load(os.path.join(self.models_dir, 'duration_model.joblib'))
        self.mappings = joblib.load(os.path.join(self.models_dir, 'mappings.joblib'))
        self.imputations = joblib.load(os.path.join(self.models_dir, 'imputations.joblib'))
        print("EISEngine: All artifacts loaded successfully.")

    def predict_event_impact(self, input_data):
        """
        input_data: dict containing:
            - event_type (str)
            - event_cause (str)
            - corridor (str)
            - police_station (str)
            - latitude (float)
            - longitude (float)
            - hour_of_day (int)
            - day_of_week (int)
            - is_weekend (int)
            - month (int)
            - junction_event_frequency (float, optional)
            - police_station_event_frequency (float, optional)
            - has_end_coords (int, optional)
            - description_length (int, optional)
            - planned_lead_time_hours (float, optional)
        """
        # Feature order must match training
        categorical_features = ['event_type', 'event_cause', 'corridor', 'police_station']
        numerical_features = [
            'latitude', 'longitude', 'hour_of_day', 'day_of_week', 'is_weekend', 
            'month', 'junction_event_frequency', 'police_station_event_frequency', 
            'has_end_coords', 'description_length', 'planned_lead_time_hours'
        ]
        
        # 1. Process categoricals
        encoded_cats = []
        for col in categorical_features:
            val = input_data.get(col, 'Unknown')
            mapping = self.mappings.get(col, {})
            # Get from mapping, fallback to Unknown, fallback to 0
            code = mapping.get(val, mapping.get('Unknown', 0))
            encoded_cats.append(code)
            
        # 2. Process numericals
        processed_nums = []
        for col in numerical_features:
            # Check if provided, fallback to imputation, fallback to 0
            default_val = self.imputations.get(col, 0.0)
            val = input_data.get(col, default_val)
            if val is None or np.isnan(val):
                val = default_val
            processed_nums.append(float(val))
            
        # Combine into features array
        features_vector = np.array(encoded_cats + processed_nums).reshape(1, -1)
        
        # 3. Model predictions
        # Priority
        priority_pred_class = self.priority_model.predict(features_vector)[0]
        # In our PureDecisionTree, predict_proba returns class 1 probability
        priority_prob = self.priority_model.predict_proba(features_vector)[0]
        
        # Road Closure
        closure_pred_class = self.closure_model.predict(features_vector)[0]
        closure_prob = self.closure_model.predict_proba(features_vector)[0]
        
        # Duration
        predicted_log_duration = self.duration_model.predict(features_vector)[0]
        predicted_duration_mins = float(np.expm1(predicted_log_duration))
        if predicted_duration_mins < 0:
            predicted_duration_mins = 0.0
            
        # 4. Compute Event Impact Score (EIS)
        # EIS (1-100) = priority_score (max 40) + closure_score (max 30) + duration_score (max 30)
        # priority_prob represents the model's confidence of it being "High"
        priority_contrib = float(priority_prob) * 40.0
        
        # closure_prob represents confidence of road closure requirement
        closure_contrib = float(closure_prob) * 30.0
        
        # duration contribution: scaled by 24 hours (1440 mins)
        duration_contrib = min(predicted_duration_mins / 1440.0, 1.0) * 30.0
        
        raw_score = priority_contrib + closure_contrib + duration_contrib
        eis_score = int(round(min(max(raw_score, 1), 100)))
        
        # 5. Determine Severity and Recommendations
        if eis_score >= 70:
            severity = "CRITICAL"
            personnel = "Large Deployment (15-20 Officers + 1 Inspector)"
            barricades = "Full barricading required for diversions"
            diversions = "Active diversions; reroute via adjacent corridors"
        elif eis_score >= 40:
            severity = "MODERATE"
            personnel = "Medium Deployment (8-15 Officers)"
            barricades = "Partial barricading at key intersections"
            diversions = "Monitor traffic; initiate micro-diversions if queue length increases"
        else:
            severity = "LOW"
            personnel = "Light Deployment (3-8 Officers for monitoring)"
            barricades = "No barricading required"
            diversions = "No diversions needed; standard flow"
            
        # 6. Generate explainability bullet points
        explanation = []
        cause = input_data.get('event_cause', 'others')
        corridor = input_data.get('corridor', 'Non-corridor')
        
        explanation.append(f"Event cause '{cause}' historically correlates with a {'high' if priority_prob > 0.5 else 'normal'} risk rating.")
        
        if closure_prob > 0.5:
            explanation.append(f"High likelihood of road closure ({closure_prob*100:.1f}%) due to event characteristics.")
        else:
            explanation.append(f"Low road closure probability ({closure_prob*100:.1f}%), indicating lane-level disruptions.")
            
        explanation.append(f"Estimated resolution time is {predicted_duration_mins:.1f} minutes ({predicted_duration_mins/60.0:.1f} hours).")
        
        if corridor != 'Non-corridor':
            explanation.append(f"Event lies on the corridor '{corridor}', elevating risk of cascade delays.")
            
        if input_data.get('junction_event_frequency', 0) > 30:
            explanation.append(f"Target location contains a known repeat traffic hotspot (frequency score: {input_data['junction_event_frequency']}).")
            
        return {
            "event_impact_score": eis_score,
            "severity_level": severity,
            "priority_prediction": "High" if priority_pred_class == 1 else "Low",
            "priority_probability": float(priority_prob),
            "road_closure_prediction": "Closure Required" if closure_pred_class == 1 else "No Closure",
            "road_closure_probability": float(closure_prob),
            "predicted_duration_minutes": predicted_duration_mins,
            "deployment_personnel": personnel,
            "barricades_recommendation": barricades,
            "diversions_recommendation": diversions,
            "explanation": explanation
        }

if __name__ == "__main__":
    # Test the engine with sample data
    engine = EISEngine()
    
    test_event = {
        "event_type": "planned",
        "event_cause": "public_event",
        "corridor": "Bellary Road 1",
        "police_station": "Cubbon Park",
        "latitude": 12.9788,
        "longitude": 77.5995,
        "hour_of_day": 18,
        "day_of_week": 4, # Friday
        "is_weekend": 0,
        "month": 3,
        "junction_event_frequency": 50.0,
        "police_station_event_frequency": 212.0,
        "has_end_coords": 0,
        "description_length": 45,
        "planned_lead_time_hours": 12.0
    }
    
    res = engine.predict_event_impact(test_event)
    print("\n--- TEST PREDICTION RESULTS ---")
    for k, v in res.items():
        if k == 'explanation':
            print("Explanation:")
            for line in v:
                print(f"  - {line}")
        else:
            print(f"{k}: {v}")
