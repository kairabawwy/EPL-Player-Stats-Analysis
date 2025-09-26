EPL Player Stats Analysis & Prediction

Overview
This project analyzes English Premier League (EPL) player statistics to uncover insights and predict player performance. It combines data analysis, visualization, and machine learning to answer questions such as:
    Who are the top goal scorers and assist providers?
    Which clubs score the most goals?
    How do player statistics like shots, assists, and appearances relate to goals?
The project culminates in a Linear Regression model that predicts goals based on key player stats.

Key Features
1. Data Cleaning & Feature Engineering
    Created new metrics: GoalsPer90, Contributions, Discipline, and GoalsPerMatch.
    Handles missing values with mean imputation to ensure clean model training.

2. Visualization
    Bar charts of total goals by club.
    Pie charts of goals by player position.
    Scatter plots to explore relationships (e.g., Shots vs Goals).
    Histograms showing goal distributions.
    Correlation heatmaps for all numeric stats and key features.

3. Machine Learning
    Linear Regression model that predicts a player’s goals based on:
        Shots on target
        Assists
        Appearances
    Evaluation metrics include:
        R² Score – proportion of goal variance explained by the model.
        Mean Squared Error (MSE) – average squared difference between predicted and actual goals.
        Coefficients – effect of each feature on goals.
        Intercept – predicted goals when all features are 0.
        
4. Visualizations for ML
    Scatter plot of predicted vs actual goals.
    Bar chart showing feature importance based on regression coefficients.
