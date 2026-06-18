# 🚦 CongestIQ: Event-Driven Congestion Intelligence Engine

CongestIQ is an AI-powered decision-support system designed to forecast, analyze, and mitigate localized traffic congestion caused by planned events (e.g., cricket matches, festivals, VIP visits) and unplanned events (e.g., vehicle breakdowns, water logging, tree falls).

This project was built using real traffic incident data from Bengaluru's **Astram** platform. To ensure complete portability and compatibility across all platforms (bypassing DLL/compiled C-extension restrictions), it features a custom **pure-Python Decision Tree and Random Forest engine** built with NumPy.

---

## ⚡ Key Value Pillars

1. **📊 Predict (Event Impact Score)**
   - Computes a composable 1–100 impact score (EIS) for any event.
   - Powered by three dedicated models: Severity Classifier, Road Closure Predictor, and Duration Regressor.
2. **🗺️ Locate (Spatial Hotspot Risk Map)**
   - Displays chronic bottlenecks and real-time incident density using Folium interactive heatmaps.
   - Pulsing hotspot markers indicate historical bottleneck junctions (e.g., Mekhri Circle, Silk Board).
3. **🚔 Prescribe (Actionable Deployment Advisor)**
   - Recommends police dispatch sizes, barricading instructions, and detours.
4. **📈 Learn (Feedback Loop & Templates)**
   - Log actual resolution times to evaluate prediction accuracy and automatically update cached Event-Type Templates (e.g., IPL Match, VIP Visit).

---

## 🛠️ Technology Stack

- **Core Logic:** Python 3.x
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Custom pure-Python decision trees (`pure_tree.py`), Joblib
- **Frontend Dashboard:** Streamlit
- **Geospatial Mapping:** Folium, `streamlit-folium`
- **Charts & Visualizations:** Plotly

---

## 📁 Repository Structure

```
CongestIQ/
├── data/
│   ├── Astram event data_anonymized...csv   # Raw dataset (8,173 records)
│   ├── processed_data.csv                   # Preprocessed data
│   └── resolution_feedback_logs.csv         # Local feedback loop database
├── models/
│   ├── priority_model.joblib                # Priority classifier
│   ├── closure_model.joblib                 # Closure predictor
│   ├── duration_model.joblib                # Duration regressor
│   ├── mappings.joblib                      # Categorical encoder mappings
│   └── imputations.joblib                   # Numerical imputations
├── pages/
│   ├── 1_Overview.py                        # Analytics overview charts
│   ├── 2_Live_Risk_Map.py                   # Heatmap and hotspot map
│   ├── 3_Event_Impact_Simulator.py          # Pre-event simulator (with templates)
│   ├── 4_Post_Event_Learning.py             # Feedback logger
│   └── 5_Deep_Dive_Analytics.py             # Model metrics & distributions
├── app.py                                   # Streamlit home landing page
├── preprocess.py                            # Data engineering pipeline
├── train_models_pure.py                     # Models training pipeline
├── pure_tree.py                             # Custom ML Decision Tree library
├── eis_engine.py                            # Composable scoring logic
└── README.md                                # Documentation (This file)
```

---

## 🚀 Setup & Execution Steps

### 1. Clone/Setup Project Directory
Ensure your files are structured as shown in the directory tree above.

### 2. Install Dependencies
Install all required libraries using pip:
```bash
pip install pandas numpy scikit-learn streamlit folium plotly joblib streamlit-folium
```

### 3. Run Preprocessing Pipeline
Clean the raw dataset and generate engineered features:
```bash
python preprocess.py
```

### 4. Train Core ML Models
Train the custom decision tree classifiers and regressors:
```bash
python train_models_pure.py
```

### 5. Launch the Dashboard
Run the Streamlit application:
```bash
python -m streamlit run app.py
```
After running, the application will open automatically in your browser at `http://localhost:8501`.

---


