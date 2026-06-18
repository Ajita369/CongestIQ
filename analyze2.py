import pandas as pd

df = pd.read_csv('data/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')
df['start_dt'] = pd.to_datetime(df['start_datetime'], errors='coerce')

# Planned event descriptions (ASCII safe)
planned = df[df['event_type'] == 'planned']
print('=== PLANNED EVENT DESCRIPTIONS (ASCII safe) ===')
for i, d in enumerate(planned['description'].dropna().head(25)):
    safe = d.encode('ascii', errors='replace').decode('ascii')[:150]
    print(f'{i+1}. {safe}')

# How many planned events have route_path?
print('\n=== PLANNED EVENTS WITH ROUTE_PATH ===')
print('Total planned:', len(planned))
print('With route_path:', planned['route_path'].notna().sum())

# Corridor x cause crosstab (top corridors only)
print('\n=== TOP 10 CORRIDORS x EVENT CAUSE (event-driven only) ===')
event_causes = ['public_event', 'procession', 'vip_movement', 'protest', 'construction', 'congestion']
event_df = df[df['event_cause'].isin(event_causes)]
ct = pd.crosstab(event_df['corridor'], event_df['event_cause'])
ct['total'] = ct.sum(axis=1)
print(ct.sort_values('total', ascending=False).head(15).to_string())

# Repeat locations (hotspots)
print('\n=== MOST FREQUENT JUNCTIONS (all events) ===')
print(df['junction'].value_counts().head(20).to_string())

# Repeat locations for event-driven
print('\n=== MOST FREQUENT JUNCTIONS (event-driven) ===')
print(event_df['junction'].value_counts().head(15).to_string())

# Police station for event-driven
print('\n=== POLICE STATIONS FOR EVENT-DRIVEN ===')
print(event_df['police_station'].value_counts().head(15).to_string())

# Sample descriptions for public_event, procession, vip_movement
for cause in ['public_event', 'procession', 'vip_movement', 'protest']:
    print(f'\n=== SAMPLE DESCRIPTIONS: {cause} ===')
    subset = df[df['event_cause'] == cause]['description'].dropna().head(10)
    for i, d in enumerate(subset):
        safe = d.encode('ascii', errors='replace').decode('ascii')[:150]
        print(f'  {i+1}. {safe}')

# Duration for planned events (end - start)
planned_dur = planned.copy()
planned_dur['end_dt'] = pd.to_datetime(planned_dur['end_datetime'], errors='coerce')
planned_dur['duration_hrs'] = (planned_dur['end_dt'] - planned_dur['start_dt']).dt.total_seconds() / 3600
valid_dur = planned_dur['duration_hrs'].dropna()
valid_dur = valid_dur[(valid_dur > 0) & (valid_dur < 24*7)]
print('\n=== PLANNED EVENT DURATION (hours) ===')
print('Count:', len(valid_dur))
print('Mean:', round(valid_dur.mean(), 1))
print('Median:', round(valid_dur.median(), 1))
