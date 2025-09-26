# Import necessary libraries
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("EPL_playerstats.csv")  # Load EPL player statistics CSV file

# Feature engineering: create new metrics for analysis
df["GoalsPer90"] = df["Goals"] / df["Appearances"] * 90         # Goals per 90 minutes
df["Contributions"] = df["Goals"] + df["Assists"]               # Total goal contributions
df["Discipline"] = df["Yellow cards"] + 2*df["Red cards"]       # Weighted discipline score
df["GoalsPerMatch"] = df["Goals"] / df["Appearances"]           # Goals per match

# Sorting and top players/clubs for quick insights
best_ratio = df.sort_values("GoalsPerMatch", ascending=False).head(10)  # Best goal per match ratio
top_scorers = df.sort_values("Goals", ascending=False).head(10)         # Top goal scorers
defense = df.groupby("Club")["Goals conceded"].sum().sort_values()       # Clubs with least goals conceded
top_assists = df.sort_values("Assists", ascending=False).head(10)        # Players with most assists
top_keepers = df.sort_values("Saves", ascending=False).head(10)          # Goalkeepers with most saves
club_goals = df.groupby("Club")["Goals"].sum().sort_values(ascending=False) # Total goals per club
avg_goals = df.groupby("Club")["Goals"].mean().sort_values(ascending=False) # Average goals per club
worst_discipline = df.sort_values("Discipline", ascending=False)          # Most undisciplined players
best_discipline = df.sort_values("Discipline", ascending=True)            # Most disciplined players

# ----------------------------- VISUALIZATIONS -----------------------------

# Bar chart: total goals by club
ax = club_goals.plot(kind="bar", figsize=(10,6), color="skyblue")
ax.set_title("Total Goals by Club", fontsize=14)
ax.set_ylabel("Goals")
ax.set_xlabel("Club")
plt.xticks(rotation=45, ha="right")  # Rotate x labels for readability
plt.tight_layout()
plt.show()

# Pie chart: goals by position
position_goals = df.groupby("Position")["Goals"].sum()
position_goals.plot(kind='pie', autopct="%1.1f%%", figsize=(6,6))
plt.title("Goals by Position")
plt.ylabel("")
plt.show()

# Scatter plot: shots vs goals
plt.scatter(df["Shots"], df["Goals"], alpha=0.5)
plt.title("Shots vs Goals")
plt.xlabel("Shots")
plt.ylabel("Goals")
plt.show()

# Histogram: distribution of goals
df["Goals"].plot(kind="hist", bins=20, edgecolor="black", figsize=(8,5))
plt.title("Distribution of Goals per Player")
plt.xlabel("Goals")
plt.show()

# Correlation heatmap for all numeric stats
numeric_df = df.select_dtypes(include=["int64","float64"])
corr = numeric_df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidth=0.5)
plt.title("Correlation Heatmap of Player Stats")
plt.show()

# Scatter plots: goals vs key stats
stats = ["Shots on target", "Assists", "Appearances"]
for stat in stats:
    plt.figure(figsize=(8,6))
    plt.scatter(df[stat], df["Goals"], alpha=0.6)
    plt.title(f"Goals vs {stat}")
    plt.xlabel(stat)
    plt.ylabel("Goals")
    plt.show()

# Correlation heatmap for selected key stats
correlation = df[["Goals", "Shots on target", "Assists", "Appearances"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------- MACHINE LEARNING -----------------------------

# Define features (X) and target (y) for regression
feature_names = ["Shots on target", "Assists", "Appearances"]
x = df[feature_names]
y = df["Goals"]

# Handle missing values by imputing with column mean
imputer = SimpleImputer(strategy="mean")
x = imputer.fit_transform(x)  # Converts DataFrame to NumPy array
x_columns = feature_names     # Save feature names for plotting coefficients

# Split dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict goals on the test set
y_pred = model.predict(x_test)

# Evaluate model performance
print("R^2 Score: ", r2_score(y_test, y_pred))               # How much variation in goals is explained
print("MSE:", mean_squared_error(y_test, y_pred))           # Average squared prediction error
print("Coefficients: ", model.coef_)                        # Effect of each feature on goals
print("Intercept: ", model.intercept_)                      # Predicted goals when features are 0

# Visualize predicted vs actual goals
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)  # Perfect prediction line
plt.xlabel("Actual Goals")
plt.ylabel("Predicted Goals")
plt.title("Actual vs Predicted Goals")
plt.show()

# Bar chart of feature importance (regression coefficients)
coef = pd.Series(model.coef_, index=x_columns)
coef.plot(kind="barh", figsize=(8,6))
plt.title("Statistic Importance for Predicting Goals")
plt.show()
