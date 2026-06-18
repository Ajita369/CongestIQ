import pandas as pd
import numpy as np
import os

def preprocess_data(input_path, output_path):
    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Drop completely empty columns
    cols_to_drop = ['map_file', 'comment', 'meta_data']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # 2. Parse timestamps
    print("Parsing timestamps...")
    df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end_datetime'], errors='coerce')
    df['created_dt'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
    df['resolved_dt'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
    
    # 3. Handle coordinate out-of-bounds for Bengaluru
    # Bengaluru bbox approx: Lat [12.8, 13.3], Lon [77.3, 77.8]
    print("Filtering coordinates...")
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Impute or flag invalid lat/lon
    lat_mask = (df['latitude'] >= 12.8) & (df['latitude'] <= 13.3)
    lon_mask = (df['longitude'] >= 77.3) & (df['longitude'] <= 77.8)
    valid_coords = lat_mask & lon_mask
    
    # Drop rows with invalid coordinates since spatial prediction needs them
    df = df[valid_coords].copy()
    
    # 4. Temporal Features
    print("Engineering temporal features...")
    df['hour_of_day'] = df['start_dt'].dt.hour
    df['day_of_week'] = df['start_dt'].dt.dayofweek # 0=Monday, 6=Sunday
    df['day_name'] = df['start_dt'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['start_dt'].dt.month
    
    # 5. Compute resolution duration in minutes
    print("Computing resolution duration...")
    df['resolution_duration_minutes'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60.0
    
    # Handle negative durations or extreme outliers (e.g. > 30 days) by setting to NaN
    # We only use rows with valid resolution time for training the duration regressor.
    invalid_duration = (df['resolution_duration_minutes'] < 0) | (df['resolution_duration_minutes'] > 60 * 24 * 30)
    df.loc[invalid_duration, 'resolution_duration_minutes'] = np.nan
    
    # 6. Spatial Features
    print("Engineering spatial features...")
    # Corridor flag
    df['corridor'] = df['corridor'].fillna('Non-corridor')
    df['is_corridor'] = (df['corridor'] != 'Non-corridor').astype(int)
    
    # Junction occurrence frequency
    df['junction'] = df['junction'].fillna('Unknown')
    junction_freq = df['junction'].value_counts().to_dict()
    # Subtract 1 so it doesn't count itself if we want, or just keep count
    df['junction_event_frequency'] = df['junction'].map(junction_freq)
    
    # Police station frequency
    df['police_station'] = df['police_station'].fillna('Unknown')
    police_freq = df['police_station'].value_counts().to_dict()
    df['police_station_event_frequency'] = df['police_station'].map(police_freq)
    
    # Road segment info (end lat/lon present)
    df['endlatitude'] = pd.to_numeric(df['endlatitude'], errors='coerce')
    df['endlongitude'] = pd.to_numeric(df['endlongitude'], errors='coerce')
    df['has_end_coords'] = ((df['endlatitude'] > 0) & (df['endlongitude'] > 0)).astype(int)
    
    # 7. Clean descriptions and texts
    df['description'] = df['description'].fillna('No description')
    df['description_length'] = df['description'].apply(lambda x: len(str(x)))
    
    # Clean priority (Low / High). Impute missing with 'Low'
    df['priority'] = df['priority'].fillna('Low')
    
    # Clean requires_road_closure (bool/str to 1/0)
    df['requires_road_closure'] = df['requires_road_closure'].astype(int)
    
    # Lead time for planned events (in hours)
    df['planned_lead_time_hours'] = (df['start_dt'] - df['created_dt']).dt.total_seconds() / 3600.0
    df.loc[df['planned_lead_time_hours'] < 0, 'planned_lead_time_hours'] = 0.0
    df['planned_lead_time_hours'] = df['planned_lead_time_hours'].fillna(0.0)
    
    # Ensure directory exists for output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    raw_path = "data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    processed_path = "data/processed_data.csv"
    preprocess_data(raw_path, processed_path)
