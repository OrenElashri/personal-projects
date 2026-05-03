import matplotlib as plt
import seaborn
import pandas as pd
import numpy as np
import plotly.express as px


df = pd.read_csv('C:/Users/Oren Elashri/Documents/Study/3rd Semester/Visualization/Exercises/Project/results.csv')

new_db=df.copy()
new_db['winner_team'] = None

for index, row in new_db.iterrows():
    if row["home_score"] > row["away_score"]:
        new_db.at[index, "winner_team"] = row["home_team"]
    elif row["away_score"] > row["home_score"]:
        new_db.at[index, "winner_team"] = row["away_team"]
    else:
        new_db.at[index, "winner_team"] = "Draw"





# Count the wins for each team
num_wins_per_team = new_db['winner_team'].value_counts()

# Get the top 10 winning teams - Without draw
top_10_winners = new_db['winner_team'].value_counts()[1:11]

# Create new data frame
top_10_winners_df = top_10_winners.to_frame()
top_10_winners_df.reset_index(inplace=True)
top_10_winners_df.columns = ["Team", "Number of Wins"]

# Create a marker for each team
marker = [
    'rgba(79, 233, 110, 0.8)',    # Brazil - Green with transparency
    'rgba(186, 12, 47, 0.8)',   # England - Red with transparency
    'rgba(0, 0, 0, 0.8)',       # Germany - Black with transparency
    'rgba(0, 167, 255, 0.8)',   # Argentina - Light blue with transparency
    'rgba(0, 32, 91, 0.8)',     # Sweden - Dark Blue with transparency
    'rgba(151, 52, 52, 0.8)',   # South Korea - Red with transparency
    'rgba(0, 104, 55, 0.8)',    # Mexico - Green with transparency
    'rgba(255, 85, 85, 0.8)',     # Hungary - Red with transparency
    'rgba(0, 85, 164, 0.8)',    # France - Blue with transparency
    'rgba(218, 218, 218, 0.8)'  # Italy - Grey with transparency
]


# Create the chart
fig = px.bar(top_10_winners_df, x="Team", y="Number of Wins", title="International Top 10 Winners since July 1872 to July 2024 - Interactive Bar Chart")

# Updating the trace with color per team
fig.update_traces(marker_color=marker)


# Display the chart
fig.show()