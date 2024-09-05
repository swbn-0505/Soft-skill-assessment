import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

input_data = pd.read_csv("soft_skills_dataset_150.csv")

X = input_data[["Communication" , "Teamwork" , "Problem-Solving" , "Leadership" , "Adaptability" , "Emotional Intelligence"]]
y = input_data[["Career Type"]]

X_trainset , X_testset , y_trainset , y_testset = train_test_split(X , y , test_size = 0.2 , random_state = 3)

model = RandomForestClassifier(n_estimators = 100 , random_state = 42)
model.fit(X_trainset , y_trainset)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as 'model.pkl'.")