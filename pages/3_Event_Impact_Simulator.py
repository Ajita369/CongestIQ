import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from eis_engine import EISEngine

st.set_page_config(
    page_title="Event Impact Simulator — CongestIQ",
    page_icon="🎯",
    layout="wide"
)

# Load EISEngine
@st.cache_resource
def load_eis_engine():
    return EISEngine(models_dir="models")

engine = load_eis_engine()

# Load processed data to populate option lists and coordinates
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    return df

df = load_data()

# ----------------- SESSION STATE FOR TEMPLATES -----------------
# Define templates dictionary
TEMPLATES = {
    "IPL Match at Chinnaswamy": {
        "event_type": "planned",
        "event_cause": "public_event",
        "corridor": "CBD 2",
        "police_station": "Cubbon Park",
        "latitude": 12.9788,
        "longitude": 77.5995,
        "hour_of_day": 18,
        "day_of_week": "Friday",
        "month": 3,
        "junction": "QueensStatueCircle",
        "description": "IPL opening Ceremony - RCB vs SRH at Chinnaswamy Stadium",
        "planned_lead_time_hours": 12.0,
        "has_end_coords": 0
    },
    "Bus Breakdown on ORR": {
        "event_type": "unplanned",
        "event_cause": "vehicle_breakdown",
        "corridor": "ORR East 2",
        "police_station": "HAL Old Airport",
        "latitude": 12.9556,
        "longitude": 77.6827,
        "hour_of_day": 8,
        "day_of_week": "Monday",
        "month": 11,
        "junction": "AyyappaTempleJunc",
        "description": "BMTC bus breakdown in middle of road near Ayyappa Temple",
        "planned_lead_time_hours": 0.0,
        "has_end_coords": 0
    },
    "VIP Convoy (Mekhri Circle)": {
        "event_type": "planned",
        "event_cause": "vip_movement",
        "corridor": "Bellary Road 1",
        "police_station": "Sadashivanagar",
        "latitude": 13.0001,
        "longitude": 77.5840,
        "hour_of_day": 10,
        "day_of_week": "Wednesday",
        "month": 2,
        "junction": "MekhriCircle",
        "description": "Road Restriction for PM Visit and VVIP convoy movement",
        "planned_lead_time_hours": 24.0,
        "has_end_coords": 1
    },
    "Hebbal Water Logging": {
        "event_type": "unplanned",
        "event_cause": "water_logging",
        "corridor": "ORR North 2",
        "police_station": "Hebbala",
        "latitude": 13.0418,
        "longitude": 77.5947,
        "hour_of_day": 19,
        "day_of_week": "Thursday",
        "month": 12,
        "junction": "HebbalFlyoverJunc",
        "description": "Heavy rain water logging with single lane traffic movement",
        "planned_lead_time_hours": 0.0,
        "has_end_coords": 0
    }
}

# Initial state setup
if 'sim_inputs' not in st.session_state:
    st.session_state['sim_inputs'] = {
        "event_type": "planned",
        "event_cause": "public_event",
        "corridor": "CBD 2",
        "police_station": "Cubbon Park",
        "latitude": 12.9788,
        "longitude": 77.5995,
        "hour_of_day": 18,
        "day_of_week": "Friday",
        "month": 3,
        "junction": "QueensStatueCircle",
        "description": "IPL opening Ceremony - RCB vs SRH at Chinnaswamy Stadium",
        "planned_lead_time_hours": 12.0,
        "has_end_coords": 0
    }

def apply_template(template_name):
    st.session_state['sim_inputs'] = TEMPLATES[template_name].copy()
    st.toast(f"Applied template: {template_name}", icon="🎯")

# Custom Styling
st.markdown("""
<style>
    .result-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
        color: #ff4b4b;
    }
    .score-box {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
    }
    .recommendation-card {
        background-color: #2c3e50;
        border-radius: 8px;
        padding: 20px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        border-left: 5px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Event Impact Simulator")
st.write("Inference and planning studio. Input hypothetical event parameters to predict the Event Impact Score (EIS), road closures, and generate response plans.")

st.write("---")

# --- TOP TEMPLATES BAR ---
st.markdown("### ⚡ Quick-Load Impact Templates")
cols_t = st.columns(len(TEMPLATES))
for idx, t_name in enumerate(TEMPLATES.keys()):
    if cols_t[idx].button(t_name, use_container_width=True):
        apply_template(t_name)
        st.rerun()

st.write("---")

# Layout
col_form, col_results = st.columns([1, 1])

# Lists of unique values for dropdowns
event_causes = sorted(list(df['event_cause'].unique()))
corridors = sorted(list(df['corridor'].unique()))
police_stations = sorted(list(df['police_station'].unique()))
# Junction list without Unknown
junctions = ['Unknown'] + sorted(list(df[df['junction'] != 'Unknown']['junction'].unique()))

day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

inputs = st.session_state['sim_inputs']

with col_form:
    st.markdown("### Configure Event Parameters")
    
    # Simple form fields that synchronize with state
    sim_event_type = st.selectbox(
        "Event Type", 
        ["planned", "unplanned"], 
        index=0 if inputs["event_type"] == "planned" else 1
    )
    
    sim_cause = st.selectbox(
        "Event Cause", 
        event_causes, 
        index=event_causes.index(inputs["event_cause"]) if inputs["event_cause"] in event_causes else 0
    )
    
    sim_corridor = st.selectbox(
        "Corridor", 
        corridors, 
        index=corridors.index(inputs["corridor"]) if inputs["corridor"] in corridors else 0
    )
    
    sim_police_station = st.selectbox(
        "Responsible Police Station", 
        police_stations, 
        index=police_stations.index(inputs["police_station"]) if inputs["police_station"] in police_stations else 0
    )
    
    # Known Junction selector
    selected_junc = st.selectbox(
        "Associated Junction (Optional)", 
        junctions, 
        index=junctions.index(inputs["junction"]) if inputs["junction"] in junctions else 0
    )
    
    # Auto-fill coordinates if junction is selected
    default_lat = inputs["latitude"]
    default_lon = inputs["longitude"]
    junc_freq = 0.0
    
    if selected_junc != 'Unknown':
        # Get coordinates from data for this junction
        junc_data = df[df['junction'] == selected_junc]
        if not junc_data.empty:
            default_lat = float(junc_data['latitude'].mean())
            default_lon = float(junc_data['longitude'].mean())
            junc_freq = float(len(junc_data))
            
    lat_col, lon_col = st.columns(2)
    sim_lat = lat_col.number_input("Latitude", value=default_lat, format="%.5f")
    sim_lon = lon_col.number_input("Longitude", value=default_lon, format="%.5f")
    
    # Time settings
    hour_col, day_col, month_col = st.columns(3)
    sim_hour = hour_col.slider("Hour of Day (UTC)", 0, 23, value=int(inputs["hour_of_day"]))
    
    days_list = list(day_mapping.keys())
    sim_day_name = day_col.selectbox(
        "Day of Week", 
        days_list, 
        index=days_list.index(inputs["day_of_week"]) if inputs["day_of_week"] in days_list else 0
    )
    
    months_list = list(month_mapping.keys())
    # Month index mapping (stored index is 1-based, list is 0-based)
    m_idx = inputs["month"] - 1
    sim_month_name = month_col.selectbox(
        "Month", 
        months_list, 
        index=m_idx if 0 <= m_idx < 12 else 0
    )
    
    # Operational controls
    lead_col, closure_col = st.columns([2, 1])
    sim_lead_time = lead_col.slider("Planned Lead Time (Hours)", 0.0, 72.0, value=float(inputs["planned_lead_time_hours"]))
    sim_has_closure = closure_col.checkbox("Road Segment Closure?", value=bool(inputs["has_end_coords"]))
    
    sim_desc = st.text_area("Event Description", value=inputs["description"])
    
    # Sync current widget state back to session_state so it is preserved
    st.session_state['sim_inputs'] = {
        "event_type": sim_event_type,
        "event_cause": sim_cause,
        "corridor": sim_corridor,
        "police_station": sim_police_station,
        "latitude": sim_lat,
        "longitude": sim_lon,
        "hour_of_day": sim_hour,
        "day_of_week": sim_day_name,
        "month": month_mapping[sim_month_name],
        "junction": selected_junc,
        "description": sim_desc,
        "planned_lead_time_hours": sim_lead_time,
        "has_end_coords": 1 if sim_has_closure else 0
    }

# Run Simulator Calculation
with col_results:
    st.markdown("### Simulated Impact & Recommendation")
    
    # Prepare data for prediction
    is_weekend_val = 1 if sim_day_name in ["Saturday", "Sunday"] else 0
    
    # Compute frequency of police station
    pol_freq = float(df[df['police_station'] == sim_police_station].shape[0])
    
    inference_dict = {
        "event_type": sim_event_type,
        "event_cause": sim_cause,
        "corridor": sim_corridor,
        "police_station": sim_police_station,
        "latitude": sim_lat,
        "longitude": sim_lon,
        "hour_of_day": sim_hour,
        "day_of_week": day_mapping[sim_day_name],
        "is_weekend": is_weekend_val,
        "month": month_mapping[sim_month_name],
        "junction_event_frequency": junc_freq,
        "police_station_event_frequency": pol_freq,
        "has_end_coords": 1 if sim_has_closure else 0,
        "description_length": len(sim_desc),
        "planned_lead_time_hours": sim_lead_time
    }
    
    # Run prediction
    try:
        pred_res = engine.predict_event_impact(inference_dict)
    except Exception as e:
        st.error(f"Inference Engine failed: {e}")
        st.stop()
        
    score = pred_res["event_impact_score"]
    severity = pred_res["severity_level"]
    
    # Color coding based on score
    if score >= 70:
        score_color = "#e74c3c" # Red
        text_color = "red"
    elif score >= 40:
        score_color = "#f39c12" # Orange
        text_color = "orange"
    else:
        score_color = "#2ecc71" # Green
        text_color = "green"
        
    # Display Score Card
    st.markdown(f"""
    <div class="score-box" style="border-top: 6px solid {score_color};">
        <div class="kpi-label">Event Impact Score (EIS)</div>
        <div class="score-value" style="color: {score_color};">{score} / 100</div>
        <div style="font-weight: bold; font-size: 1.2rem; color: {score_color}; margin-top: 5px;">
            ⚠️ {severity} SEVERITY LEVEL
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Secondary Indicators
    col_ind1, col_ind2 = st.columns(2)
    
    # closure formatting
    prob_closure = pred_res["road_closure_probability"]
    col_ind1.metric(
        "Closure Probability", 
        f"{prob_closure*100:.1f}%", 
        delta="High Probability" if prob_closure > 0.5 else "Low Probability",
        delta_color="inverse"
    )
    
    # duration formatting
    dur_mins = pred_res["predicted_duration_minutes"]
    col_ind2.metric(
        "Expected Recovery", 
        f"{dur_mins/60.0:.1f} Hours", 
        f"{dur_mins:.0f} Mins"
    )
    
    # Actionable Prescription
    st.write("---")
    st.markdown("<div class='result-header'>🚔 Prescribed Deployment Plan</div>", unsafe_allow_html=True)
    
    card_border_color = score_color
    st.markdown(f"""
    <div class="recommendation-card" style="border-left: 6px solid {card_border_color};">
        <b>🚨 Dispatch Recommendation:</b> {pred_res['deployment_personnel']}<br><br>
        <b>🚧 Barricading Status:</b> {pred_res['barricades_recommendation']}<br><br>
        <b>🔀 Detour Routing:</b> {pred_res['diversions_recommendation']}
    </div>
    """, unsafe_allow_html=True)
    
    # Explainability Bullet points
    st.markdown("#### Explainable AI (XAI) Logs:")
    for line in pred_res['explanation']:
        st.write(f"- {line}")
        
    # Render mini-map of event impact zone
    st.write("---")
    st.markdown("#### Predicted Impact Zone Map")
    
    # Create Folium Map centered on simulated location
    map_radius = min(max(score * 10, 100), 1000) # Circle radius based on score (100m - 1000m)
    
    mini_m = folium.Map(location=[sim_lat, sim_lon], zoom_start=14, tiles="cartodbpositron")
    
    # Circle of predicted impact
    folium.Circle(
        location=[sim_lat, sim_lon],
        radius=map_radius,
        popup=f"Simulated Event Zone<br>Predicted Impact Radius: {map_radius}m",
        color=score_color,
        fill=True,
        fill_color=score_color,
        fill_opacity=0.3
    ).add_to(mini_m)
    
    # Event marker
    folium.Marker(
        location=[sim_lat, sim_lon],
        popup=f"<b>Simulated Event</b><br>Cause: {sim_cause}<br>EIS: {score}",
        icon=folium.Icon(color="red" if score >= 70 else "blue", icon="warning-sign")
    ).add_to(mini_m)
    
    # Plotly/Folium render
    st_folium(mini_m, width="100%", height=300, key="simulator_map", returned_objects=[])
