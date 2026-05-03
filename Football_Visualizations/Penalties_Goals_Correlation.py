import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load both the data
df1 = pd.read_csv('C:/Users/Oren Elashri/Documents/Study/3rd Semester/Visualization/Exercises/Project/goalscorers.csv')
df2 = pd.read_csv('C:/Users/Oren Elashri/Documents/Study/3rd Semester/Visualization/Exercises/Project/results.csv')


# Create game detailed column in both of the df 
df1['game'] = list(zip(df1['date'],df1['home_team'], df1['away_team']))
df2['game'] = list(zip(df2['date'],df2['home_team'], df2['away_team']))

# Merged df1 and df2
m = df1.merge(df2,on='game')

# Calculate goals per game
m['total_goals'] = 1

# Details from group by Tournament
data_group_by_tournament = m.groupby('tournament').agg(
    total_penalties = ('penalty', lambda x: x.sum()),
    total_games = ('game', 'nunique'),
    total_goals = ('total_goals', 'sum')
).reset_index()

# Calculate the average of penalties per game, the average of goals per game, the average of penalties per game divided by the average of goals per game
data_group_by_tournament['avg_penalties_game'] = data_group_by_tournament['total_penalties'] / data_group_by_tournament['total_games']
data_group_by_tournament['avg_goals_game'] = data_group_by_tournament['total_goals'] / data_group_by_tournament['total_games']
data_group_by_tournament['relation_penalties_goals_game'] = data_group_by_tournament['avg_penalties_game'] / data_group_by_tournament['avg_goals_game']


# Senity check Total Num of Penalties, Games and Goals
total_num_penalties = np.sum(data_group_by_tournament['total_penalties'])
total_num_games = np.sum(data_group_by_tournament['total_games'])
total_num_goals = len(m)

# Select relevant columns for the heatmap
last_df = data_group_by_tournament[['tournament', 'avg_penalties_game', 'avg_goals_game', 'relation_penalties_goals_game']]

# Transform for heatmap visualization
heatmap_df = last_df.set_index('tournament').T



# Plotting-  Create subplots with one heatmap per row
fig = make_subplots(
    rows=heatmap_df.shape[0],
    cols=1,
    subplot_titles=heatmap_df.index
)
colormaps = ["Viridis", "Cividis", "Bluered"]


for i, row in enumerate(heatmap_df.index):
    colorscale = colormaps[i % len(colormaps)]
    coloraxis_name = f"coloraxis{i + 1}"  # Unique coloraxis for each subplot
    fig.add_trace(
        go.Heatmap(
            z=[heatmap_df.loc[row].values],
            x=heatmap_df.columns,
            y=[row],
            colorscale=colorscale,
            colorbar=dict(
                title=row,  # Use the row name as the colorbar title
                len=0.8,  # Adjust colorbar height
                y=0.85 - (i * (1 / heatmap_df.shape[0])),  # Dynamically position the colorbar vertically
                yanchor="middle",  # Center the colorbar vertically with the subplot
                x=1.05 + i * 0.05,  # Position the colorbar to the right of the subplot
                xanchor="left"  # Align the left edge of the colorbar with x
            )
        ),        row=i + 1,
        col=1
    )
    for j, x_val in enumerate(heatmap_df.columns):
        value = heatmap_df.loc[row, x_val]
        fig.add_annotation(
            x=x_val,
            y=row,
            text=f"{value:.2f}",  # Format the value (adjust as needed)
            showarrow=False,  # Disable arrows
            font=dict(size=12, color="white"),  # Customize font style
            xref="x",
            yref="y",
            row=i + 1,
            col=1
        )
    fig.update_layout({coloraxis_name: dict(colorscale="Viridis")})  # Update each coloraxis

fig.update_layout(
    title="Realation betwenn Penalties and Goals per Game per Tournament since July 1916 - Interactive Heatmap",
    height=400 * heatmap_df.shape[0]  # Adjust height dynamically
)
fig.show()

