import pandas as pd
import plotly.express as px


# Load the data
df = pd.read_csv('C:/Users/Oren Elashri/Documents/Study/3rd Semester/Visualization/Exercises/Project/goalscorers.csv')

# Concvert the date column to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')

# Extract the decade (### - ###0) from the date
df['decade'] = (df['date'].dt.year // 10) * 10

# Create game detailed column 
df['game'] = list(zip(df['date'],df['home_team'], df['away_team']))

# Group decade by game and then take the mean of own_goal per game
game = df.groupby(['decade','game'])['own_goal'].mean()
game = game.reset_index()

#mean of own goals by decade by game
mean_own_goal_per_decade = game.groupby('decade')['own_goal'].mean().reset_index()
print(df)

# Plotting
fig = px.line(mean_own_goal_per_decade, x="decade", y="own_goal", title="Average Own Goals per Decade since July 1916 - Interactive Line Chart")
fig.update_traces(line=dict(color='black', width=3))
fig.show()