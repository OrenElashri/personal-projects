import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load both the data
df = pd.read_csv('C:/Users/Oren Elashri/Documents/Study/3rd Semester/Visualization/Exercises/Project/goalscorers.csv')

# Convert the date column to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')

# Extract the decade (### - ###0) from the date
df['decade'] = (df['date'].dt.year // 10) * 10


df['total_goals_per_scorer'] = 1

# Details from group by decade and scorer
data_group_by_decade_scorer = df.groupby(['decade','scorer']).agg(
    total_goals_per_scorer = ('total_goals_per_scorer', 'sum'),
).reset_index()


# Top 5 scorers per decade
top_5_per_decade = (data_group_by_decade_scorer.sort_values(['decade','total_goals_per_scorer'], ascending = [True, False])
                    .groupby('decade')
                    .head(5)
                    .reset_index(drop=True))

for decade in sorted(top_5_per_decade['decade'].unique()):
    # print(f"\nTop 5 Scorers in {decade}s:")
    decade_data = top_5_per_decade[top_5_per_decade['decade'] == decade]
    # print(decade_data[['scorer', 'total_goals_per_scorer']].to_string(index=False))


# Ensure the scale for goals is consistent across decades
goal_min = top_5_per_decade['total_goals_per_scorer'].min()
goal_max = top_5_per_decade['total_goals_per_scorer'].max()

# Create a list of dimensions for parallel coordinates
dimensions = [
    dict(
        label=f"{decade}s",
        values=top_5_per_decade[top_5_per_decade['decade'] == decade]['total_goals_per_scorer'],
        range=[goal_min, goal_max],
        ticktext=[f"{scorer} ({goals})" 
                 for scorer, goals in zip(
                     top_5_per_decade[top_5_per_decade['decade'] == decade]['scorer'],
                     top_5_per_decade[top_5_per_decade['decade'] == decade]['total_goals_per_scorer']
                 )],
        tickvals=top_5_per_decade[top_5_per_decade['decade'] == decade]['total_goals_per_scorer']
    )
    for decade in sorted(top_5_per_decade['decade'].unique())
]

# Plot using Plotly Graph Objects
fig = go.Figure(data=go.Parcoords(
    line=dict(color=top_5_per_decade['total_goals_per_scorer'],
              colorscale='Viridis'),
    dimensions=dimensions
))

# Update layout for better visualization
fig.update_layout(title="Top 5 Goal Scorers Per Decade in Parallel Coordinates")
fig.show()