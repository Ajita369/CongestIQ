import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="City Traffic Overview — CongestIQ",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        background-color: #1e272e;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        border-top: 4px solid #ff4b4b;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f5f6fa;
        margin-top: 5px;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    df['start_dt'] = pd.to_datetime(df['start_dt'], errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_dt'], errors='coerce')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.title("📊 City Traffic Overview")
st.write("Real-time aggregated operational metrics and trends based on 8,173 reported incidents across Bengaluru.")

st.write("---")

# 1. KPI Cards Row
total_events = len(df)
# Valid durations only for mean
valid_durations = df['resolution_duration_minutes'].dropna()
avg_res_hours = valid_durations.mean() / 60.0 if len(valid_durations) > 0 else 0.0
total_closures = df['requires_road_closure'].sum()
# Event driven subset
event_causes = ['public_event', 'procession', 'vip_movement', 'protest', 'construction', 'congestion']
total_event_driven = df[df['event_cause'].isin(event_causes)].shape[0]

kpi_html = f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-label">Total Incidents</div>
        <div class="kpi-val">{total_events:,}</div>
    </div>
    <div class="kpi-card" style="border-top: 4px solid #3498db;">
        <div class="kpi-label">Avg Resolution Time</div>
        <div class="kpi-val">{avg_res_hours:.1f} Hrs</div>
    </div>
    <div class="kpi-card" style="border-top: 4px solid #e74c3c;">
        <div class="kpi-label">Road Closures</div>
        <div class="kpi-val">{total_closures:,}</div>
    </div>
    <div class="kpi-card" style="border-top: 4px solid #2ecc71;">
        <div class="kpi-label">Event-Driven</div>
        <div class="kpi-val">{total_event_driven:,}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# 2. Charts Layout (Row 1)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Incident Distribution by Cause")
    # Donut Chart for Event Causes
    cause_counts = df['event_cause'].value_counts().reset_index()
    cause_counts.columns = ['Cause', 'Count']
    
    fig_donut = px.pie(
        cause_counts, 
        values='Count', 
        names='Cause', 
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark"
    )
    fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.markdown("### Top 15 Hotspot Corridors")
    # Corridor Horizontal Bar Chart
    corridor_counts = df['corridor'].value_counts().head(15).reset_index()
    corridor_counts.columns = ['Corridor', 'Incidents']
    # Sort for visual order
    corridor_counts = corridor_counts.sort_values('Incidents', ascending=True)
    
    fig_bar = px.bar(
        corridor_counts, 
        x='Incidents', 
        y='Corridor', 
        orientation='h',
        color='Incidents',
        color_continuous_scale='Reds',
        template="plotly_dark"
    )
    fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

st.write("---")

# 3. Charts Layout (Row 2)
col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("### Temporal Heatmap: Hour vs Day of Week")
    
    # Heatmap calculation
    # Reindex day names to ensure correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Let's map day of week int to names or use start_dt day name
    df['day_name_val'] = df['start_dt'].dt.day_name()
    
    heatmap_df = df.groupby(['day_name_val', 'hour_of_day']).size().reset_index(name='count')
    
    # Pivot
    pivot_df = heatmap_df.pivot(index='day_name_val', columns='hour_of_day', values='count').fillna(0)
    # Reorder index
    pivot_df = pivot_df.reindex(days_order)
    
    fig_heatmap = px.imshow(
        pivot_df.values,
        labels=dict(x="Hour of Day (UTC)", y="Day of Week", color="Incidents"),
        x=list(pivot_df.columns),
        y=list(pivot_df.index),
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )
    fig_heatmap.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col4:
    st.markdown("### Monthly Trend by Event Type")
    
    # Parse month name/year for sorting
    monthly_df = df.copy()
    monthly_df['Month-Year'] = monthly_df['start_dt'].dt.to_period('M').astype(str)
    
    trend_df = monthly_df.groupby(['Month-Year', 'event_type']).size().reset_index(name='Incidents')
    trend_df = trend_df.sort_values('Month-Year')
    
    fig_area = px.area(
        trend_df, 
        x='Month-Year', 
        y='Incidents', 
        color='event_type',
        color_discrete_sequence=['#ff4b4b', '#3498db'],
        template="plotly_dark"
    )
    fig_area.update_layout(margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_area, use_container_width=True)
