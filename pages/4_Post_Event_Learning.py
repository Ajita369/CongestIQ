import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Post-Event Learning — CongestIQ",
    page_icon="📈",
    layout="wide"
)

# Custom css
st.markdown("""
<style>
    .template-card {
        background-color: #1e272e;
        border-radius: 8px;
        padding: 15px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        border-top: 4px solid #3498db;
    }
    .template-title {
        font-weight: bold;
        color: #3498db;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_data.csv")
    df['start_dt'] = pd.to_datetime(df['start_dt'], errors='coerce')
    return df

df = load_data()

st.title("📈 Post-Event Learning & Feedback Loop")
st.write("Closing the planning loop. Capture actual traffic resolution outcomes to build institutional memory and audit model accuracy.")

st.write("---")

col_form, col_templates = st.columns([1, 1])

# Path for saving feedback logs
feedback_file = "data/resolution_feedback_logs.csv"

# Handle feedback submission
with col_form:
    st.markdown("### Log Actual Event Outcomes")
    st.write("Enter details for a recently resolved event to log it in our feedback database.")
    
    # Select from active/recent incidents in the database
    # Show top 50 recent events as options
    recent_events = df.sort_values('start_dt', ascending=False).head(50)
    event_options = [f"{row['id']} - {row['event_cause']} at {str(row['address'])[:50]}..." for idx, row in recent_events.iterrows()]
    
    selected_event_str = st.selectbox("Select Resolved Event ID", event_options)
    selected_event_id = selected_event_str.split(" ")[0]
    
    # Form inputs for actual outcome
    actual_duration = st.number_input("Actual Resolution Duration (Minutes)", min_value=1, value=60)
    actual_closure = st.selectbox("Was a Road Closure Required?", ["No", "Yes"])
    deployment_adequacy = st.selectbox(
        "Resource Deployment Adequacy", 
        [
            "Sufficient (Right-sized deployment)", 
            "Under-staffed (Needed more personnel/barricades)", 
            "Over-staffed (Wasted personnel resources)"
        ]
    )
    operational_notes = st.text_area("Operational Notes", placeholder="E.g., crane arrived late, rain compounded congestion, etc.")
    
    if st.button("Submit Resolution Log"):
        # Create log entry
        new_log = pd.DataFrame([{
            "event_id": selected_event_id,
            "actual_duration_mins": actual_duration,
            "actual_closure": 1 if actual_closure == "Yes" else 0,
            "deployment_adequacy": deployment_adequacy,
            "notes": operational_notes,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        # Save to CSV
        if os.path.exists(feedback_file):
            # Check if headers match or file is empty
            try:
                new_log.to_csv(feedback_file, mode='a', header=False, index=False)
            except Exception as e:
                new_log.to_csv(feedback_file, index=False)
        else:
            new_log.to_csv(feedback_file, index=False)
            
        st.success(f"Feedback successfully logged for event **{selected_event_id}**!")
        st.balloons()

# Display historical averages by cause for templates
with col_templates:
    st.markdown("### 📋 Reusable Event-Type Templates")
    st.write("Pre-computed baseline statistics derived directly from the Bengaluru traffic incident database:")
    
    # We will compute statistics for main causes
    causes_to_show = ['public_event', 'procession', 'vip_movement', 'construction', 'water_logging']
    
    for cause in causes_to_show:
        cause_df = df[df['event_cause'] == cause]
        avg_dur = cause_df['resolution_duration_minutes'].mean()
        avg_dur_str = f"{avg_dur:.1f} Mins" if avg_dur < 120 else f"{avg_dur/60.0:.1f} Hours"
        closure_rate = cause_df['requires_road_closure'].mean() * 100
        total_events = len(cause_df)
        
        # Determine recommended baseline officers
        if cause == 'vip_movement':
            officers = "15-20 Officers (High closure rate)"
            color = "#e74c3c"
        elif cause in ['public_event', 'procession']:
            officers = "10-15 Officers"
            color = "#f39c12"
        elif cause == 'construction':
            officers = "5-8 Officers (Long-term monitoring)"
            color = "#3498db"
        else:
            officers = "4-6 Officers"
            color = "#2ecc71"
            
        st.markdown(f"""
        <div class="template-card" style="border-top: 4px solid {color};">
            <div class="template-title" style="color: {color};">{cause.upper().replace('_', ' ')} TEMPLATE</div>
            <b>Incident Count:</b> {total_events:,} historical incidents<br>
            <b>Avg Recovery Time:</b> {avg_dur_str}<br>
            <b>Road Closure Rate:</b> {closure_rate:.1f}%<br>
            <b>Recommended Dispatch:</b> {officers}
        </div>
        """, unsafe_allow_html=True)

st.write("---")

st.markdown("### 📊 Learning Audit Trail")
# Load and show logged feedbacks
if os.path.exists(feedback_file):
    try:
        feedback_df = pd.read_csv(feedback_file)
        if not feedback_df.empty:
            st.markdown(f"**Total Logs Logged:** `{len(feedback_df)}` entries")
            
            # Display logs dataframe
            st.dataframe(feedback_df, use_container_width=True)
            
            # Visual comparison chart if logs have entries
            fig = px.bar(
                feedback_df, 
                x='event_id', 
                y='actual_duration_mins', 
                color='deployment_adequacy',
                title="Actual Resolution Duration by Logged Event",
                labels={'event_id': 'Event ID', 'actual_duration_mins': 'Duration (Mins)'},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback entries logged yet. Submit the form above to add a log entry.")
    except Exception as e:
        st.warning(f"Error loading audit trail database: {e}")
else:
    # Show mock visual to represent what will happen
    st.info("No feedback entries logged yet. Submit the form above to start building the learning database.")
    
    # Simulated comparison data
    mock_data = pd.DataFrame({
        'Event ID': ['FKID0012', 'FKID0145', 'FKID0254', 'FKID0312'],
        'Predicted Duration (Mins)': [65, 45, 120, 180],
        'Actual Duration (Mins)': [60, 48, 145, 172]
    })
    
    fig_mock = px.bar(
        mock_data.melt(id_vars='Event ID'),
        x='Event ID',
        y='value',
        color='variable',
        barmode='group',
        title="Accuracy Audit (Comparison Example)",
        labels={'value': 'Duration (Mins)', 'variable': 'Metric'},
        color_discrete_sequence=['#3498db', '#2ecc71'],
        template="plotly_dark"
    )
    st.plotly_chart(fig_mock, use_container_width=True)
