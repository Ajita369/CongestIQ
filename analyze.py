import pandas as pd

df = pd.read_csv('data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# Parse dates
df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['end_dt'] = pd.to_datetime(df['end_datetime'], errors='coerce')
df['created_dt'] = pd.to_datetime(df['created_date'], errors='coerce')
df['closed_dt'] = pd.to_datetime(df['closed_datetime'], errors='coerce')

# Date range
print('=== DATE RANGE ===')
print('Start:', df['start_dt'].min(), 'to', df['start_dt'].max())
print('Created:', df['created_dt'].min(), 'to', df['created_dt'].max())

# Resolution time
df['resolution_mins'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60
valid_res = df['resolution_mins'].dropna()
valid_res = valid_res[(valid_res > 0) & (valid_res < 60*24*30)]
print('\n=== RESOLUTION TIME (minutes) ===')
print('Count:', len(valid_res))
print('Mean:', round(valid_res.mean(), 1))
print('Median:', round(valid_res.median(), 1))
print('25th:', round(valid_res.quantile(0.25), 1))
print('75th:', round(valid_res.quantile(0.75), 1))

# By event cause
print('\n=== RESOLUTION TIME BY CAUSE ===')
df_valid = df[(df['resolution_mins'] > 0) & (df['resolution_mins'] < 60*24*30)]
print(df_valid.groupby('event_cause')['resolution_mins'].agg(['count', 'mean', 'median']).sort_values('count', ascending=False).to_string())

# Hour distribution
print('\n=== HOUR OF DAY DISTRIBUTION ===')
print(df['start_dt'].dt.hour.value_counts().sort_index().to_string())

# Day of week
print('\n=== DAY OF WEEK ===')
print(df['start_dt'].dt.day_name().value_counts().to_string())

# Monthly
print('\n=== MONTHLY ===')
print(df['start_dt'].dt.to_period('M').value_counts().sort_index().to_string())

# Planned events detail
planned = df[df['event_type'] == 'planned']
print('\n=== PLANNED EVENTS - event_cause ===')
print(planned['event_cause'].value_counts().to_string())
print('\nPlanned events with end_datetime:', planned['end_dt'].notna().sum())

# Event-driven causes
event_driven = df[df['event_cause'].isin(['public_event', 'procession', 'vip_movement', 'protest', 'construction', 'congestion'])]
print('\n=== EVENT-DRIVEN SUBSET:', len(event_driven), 'rows ===')
print(event_driven['event_cause'].value_counts().to_string())
print('\nPlanned vs Unplanned in event-driven:')
print(event_driven['event_type'].value_counts().to_string())

# Road closure by cause
print('\n=== ROAD CLOSURE RATE BY CAUSE ===')
print(df.groupby('event_cause')['requires_road_closure'].mean().sort_values(ascending=False).to_string())

# Geo bounds
print('\n=== GEO BOUNDS ===')
print('Lat:', round(df['latitude'].min(), 4), 'to', round(df['latitude'].max(), 4))
print('Lon:', round(df['longitude'].min(), 4), 'to', round(df['longitude'].max(), 4))

# Priority by cause
print('\n=== PRIORITY DISTRIBUTION BY CAUSE ===')
print(pd.crosstab(df['event_cause'], df['priority']).to_string())

# Corridor frequency for event-driven
print('\n=== TOP CORRIDORS FOR EVENT-DRIVEN ===')
print(event_driven['corridor'].value_counts().head(15).to_string())

# Some planned event descriptions
print('\n=== SAMPLE PLANNED EVENT DESCRIPTIONS ===')
planned_desc = planned['description'].dropna().head(20)
for i, d in enumerate(planned_desc):
    print(f'{i+1}. {d[:120]}')
