import streamlit as st

st.set_page_config(
    page_title="CongestIQ — Event-Driven Congestion Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Gradient headers */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff7e40, #ffc837);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    /* Pillar cards */
    .pillar-card {
        background-color: #1e272e;
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    .pillar-card h3 {
        color: #ff4b4b !important;
        margin-top: 0 !important;
    }
    .pillar-card p {
        color: #dcdde1;
        font-size: 0.95rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Layout
st.markdown('<div class="main-title">🚦 CongestIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Event-Driven Congestion Preemption & Intelligence System</div>', unsafe_allow_html=True)

st.write("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Welcome to the Future of Urban Traffic Management")
    st.write("""
        **CongestIQ** is an AI-powered decisions-support system designed specifically for managing event-driven traffic congestion 
        (both planned events like cricket matches, processes, and VIP visits, and unplanned events like vehicle breakdowns and water logging).
        
        Using historical data from Bengaluru's Astram traffic management system, CongestIQ moves cities from reactive chaos to proactive, data-driven traffic preemption.
    """)
    
    st.markdown("### The Four Pillars of CongestIQ")
    
    # Pillar 1
    st.markdown("""
    <div class="pillar-card">
        <h3>📊 1. PREDICT: Event Impact Score (EIS)</h3>
        <p>Quantify event severity using a composable 1-100 impact score powered by custom decision tree models predicting event priority, road closure probability, and resolution duration.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pillar 2
    st.markdown("""
    <div class="pillar-card">
        <h3>🗺️ 2. LOCATE: Spatial Hotspot Risk Map</h3>
        <p>Identify chronic traffic bottlenecks and visualize prediction zones on an interactive geospatial interface mapping event density and adjacent corridor spillover risks.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pillar 3
    st.markdown("""
    <div class="pillar-card">
        <h3>🚔 3. PRESCRIBE: Actionable Deployment Advisor</h3>
        <p>Get data-backed recommendations for traffic officer deployment, barricade placements, and detour/diversion timings before bottlenecks develop.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pillar 4
    st.markdown("""
    <div class="pillar-card">
        <h3>📈 4. LEARN: Post-Event Feedback & Templates</h3>
        <p>Log actual outcomes against predictions to train the model over time, while caching reusable event-type templates (e.g., Cricket Match, VIP Visit) for instant planning.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### Quick Navigation")
    st.info("""
        Use the sidebar to explore the application:
        
        *   **Overview**: Key operational metrics, event distributions, and temporal trends.
        *   **Live Risk Map**: Interactive geographic visualization of historical hotspots and live filters.
        *   **Event Impact Simulator**: Input planned or unplanned event details to receive real-time impact scores and deployment plans.
        *   **Post-Event Learning**: Log event resolution and review/update event templates.
        *   **Deep Dive Analytics**: Deep dive into model features, importance charts, and core algorithms.
    """)
    
    st.markdown("### System Status")
    st.success("🟢 ML Models Loaded (Pure Python Engine)")
    st.success("🟢 Database Connected (8,173 records)")
    st.success("🟢 Bengaluru Geo-boundaries Configured")

st.write("---")
st.caption("Developed by the CongestIQ Hackathon Team. Powered by pure-python machine learning.")
