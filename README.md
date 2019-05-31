# Predicting survival on the Titanic with Python

Solving the Kaggle challenge "Titanic: Machine Learning from Disaster".
https://www.kaggle.com/c/titanic

After getting ~80% prediction score with my own model (self implementation of AdaBoost with
Decision Stumps), I changed the code to the solution made by LD Freeman (which can be found at
https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy), which uses
Decision Trees with RFE and CV along with GridSearchCV.

This code contains only one of the possible solutions suggested by LD Freeman, yet many of the code
written here can be used for other solutions with little to no changes.

The current solution gets ~87% train score, and ~82% validation score.