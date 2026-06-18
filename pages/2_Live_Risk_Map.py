import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Live Risk Map — CongestIQ",
    page_icon="🗺️",
    layout="wide"
)

# Custom css for map styling
st.markdown("""
<style>
    .legend-container {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 15px;
        color: white;
        margin-top: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .legend-title {
        font-weight: bold;
        color: #ff4b4b;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    df['start_dt'] = pd.to_datetime(df['start_dt'], errors='coerce')
    return df

df = load_data()

st.title("🗺️ Live Risk Map")
st.write("Geospatial density and bottleneck identification engine. Filter historical traffic incidents and analyze chronic bottleneck areas.")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Configuration")

# Event Type filter
event_types = ['All'] + list(df['event_type'].unique())
selected_type = st.sidebar.selectbox("Event Type", event_types)

# Event Cause filter
causes = ['All'] + sorted(list(df['event_cause'].unique()))
selected_cause = st.sidebar.selectbox("Event Cause", causes)

# Priority filter
priorities = ['All'] + list(df['priority'].unique())
selected_priority = st.sidebar.selectbox("Priority Level", priorities)

# Corridor filter
corridors = ['All'] + sorted(list(df['corridor'].unique()))
selected_corridor = st.sidebar.selectbox("Corridor Filter", corridors)

# Map Layer Controls
st.sidebar.subheader("Map Visual Layers")
show_heatmap = st.sidebar.checkbox("Show Heatmap Overlay", value=True)
show_markers = st.sidebar.checkbox("Show Clustered Markers", value=False)
show_hotspots = st.sidebar.checkbox("Show Top 20 Hotspots", value=True)

map_theme = st.sidebar.selectbox("Map Theme Style", ["Dark Mode (CartoDB Dark)", "Light Mode (CartoDB Positron)", "Standard (OpenStreetMap)"])

# Translate style name
tiles = "cartodbdark_matter"
if "Light" in map_theme:
    tiles = "cartodbpositron"
elif "Standard" in map_theme:
    tiles = "OpenStreetMap"

# Apply filters to data
filtered_df = df.copy()

if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['event_type'] == selected_type]
if selected_cause != 'All':
    filtered_df = filtered_df[filtered_df['event_cause'] == selected_cause]
if selected_priority != 'All':
    filtered_df = filtered_df[filtered_df['priority'] == selected_priority]
if selected_corridor != 'All':
    filtered_df = filtered_df[filtered_df['corridor'] == selected_corridor]

st.markdown(f"**Filtered Events Count:** `{len(filtered_df):,}` / `{len(df):,}`")

# Set map center on average coordinates
if not filtered_df.empty:
    center_lat = filtered_df['latitude'].mean()
    center_lon = filtered_df['longitude'].mean()
else:
    center_lat = 12.9788
    center_lon = 77.5995

# Initialize Map
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=tiles)

# 1. Add Heatmap
if show_heatmap and not filtered_df.empty:
    # Heatmap expects list of [lat, lon, weight]
    heat_data = filtered_df[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

# 2. Add Top 20 Hotspots
if show_hotspots:
    # Identify top 20 junctions
    hotspot_junctions = df[df['junction'] != 'Unknown']['junction'].value_counts().head(20).index.tolist()
    
    # Calculate average coords of top junctions
    for junc in hotspot_junctions:
        junc_df = df[df['junction'] == junc]
        lat = junc_df['latitude'].mean()
        lon = junc_df['longitude'].mean()
        cnt = len(junc_df)
        top_cause = junc_df['event_cause'].value_counts().index[0]
        
        # Add special glowing red/orange circle for top hotspots
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + (cnt / 15.0), # Size based on count
            popup=f"<b>Hotspot:</b> {junc}<br><b>Incidents:</b> {cnt}<br><b>Primary Cause:</b> {top_cause}",
            color="#ff3333",
            fill=True,
            fill_color="#ff5555",
            fill_opacity=0.6,
            tooltip=f"{junc} ({cnt} events)"
        ).add_to(m)

# 3. Add Clustered Individual Markers
if show_markers and not filtered_df.empty:
    marker_cluster = MarkerCluster(name="Individual Incidents").add_to(m)
    
    # Limit markers to max 1000 for performance
    display_limit = min(len(filtered_df), 1000)
    sample_df = filtered_df.sample(display_limit, random_state=42) if len(filtered_df) > 1000 else filtered_df
    
    for idx, row in sample_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        cause = row['event_cause']
        priority = row['priority']
        address = row['address']
        
        icon_color = "red" if priority == "High" else "blue"
        popup_text = f"""
        <b>Cause:</b> {cause}<br>
        <b>Priority:</b> {priority}<br>
        <b>Address:</b> {address[:100]}...
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
            tooltip=f"{cause} ({priority})"
        ).add_to(marker_cluster)

# Render Map
col_map, col_details = st.columns([3, 1])

with col_map:
    st_folium(m, width="100%", height=550, returned_objects=[])

with col_details:
    st.markdown("### Hotspot Analysis")
    st.write("""
        This map integrates structural bottleneck data with dynamic incident density:
        - **Red Glowing Circles**: Highlight top-20 structural hotspots. Hover to view event frequency and main congestion driver.
        - **Heatmap Overlay**: Identifies localized congestion zones. Dense red areas denote high frequency of unplanned vehicle breakdowns or pothole complaints.
    """)
    
    # Summary Table of Top Hotspots
    st.markdown("#### Top 5 Chronic Bottlenecks")
    top_5 = df[df['junction'] != 'Unknown']['junction'].value_counts().head(5).reset_index()
    top_5.columns = ['Junction', 'Incident Count']
    st.dataframe(top_5, hide_index=True)
    
    st.markdown("""
    <div class="legend-container">
        <div class="legend-title">Legend</div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: #ff3333; border-radius: 50%; margin-right: 8px;"></div>
            <span>Chronic Hotspot Junction</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background: linear-gradient(to right, blue, green, yellow, red); margin-right: 8px;"></div>
            <span>Incident Density (Heatmap)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 15px; height: 15px; background-color: #3498db; border-radius: 3px; margin-right: 8px; text-align:center; color:white; font-size:10px; line-height:15px;">i</div>
            <span>Individual Event Marker</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
