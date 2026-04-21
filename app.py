import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("F1_data.csv")

winner_le = LabelEncoder()
team_le = LabelEncoder()
gp_le = LabelEncoder()

data['Winner_encoded'] = winner_le.fit_transform(data['Winner'])
data['Teams_encoded'] = team_le.fit_transform(data['Teams'])
data['GP_encoded'] = gp_le.fit_transform(data['Grand_prix'])

X = data[['Teams_encoded','GP_encoded','Year']]
y = data['Winner_encoded']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

st.title("Who wins the Grand Prix?")

team = st.selectbox("Select Team", data['Teams'].unique())
gp = st.selectbox("Select Grand Prix", data['Grand_prix'].unique())
year = st.number_input("Enter Year", step=1)

if st.button("Predict Winner"):

    # Encode input
    team_encoded = team_le.transform([team])[0]
    gp_encoded = gp_le.transform([gp])[0]

    # Predict
    prediction = model.predict([[team_encoded, gp_encoded, year]])
    winner = winner_le.inverse_transform(prediction)

    # Show winner
    st.success(f"🏆 Predicted Winner: {winner[0]}")

    # Show image
import os
import re

driver_name = winner[0]
driver_name = re.sub(r'[^a-zA-Z]', '', driver_name)

base_path = "drivers"

# check all files
files = os.listdir(base_path)

found_image = None

for file in files:
    cleaned_file = re.sub(r'[^a-zA-Z]', '', file)
    
    if driver_name.lower() in cleaned_file.lower():
        found_image = os.path.join(base_path, file)
        break

if found_image:
    st.image(found_image, width=250)
else:
    st.error("Driver image not found")


    # Description
    st.write(
        f"Based on historical Formula 1 race data, the model predicts that {winner[0]} "
        f"is likely to win the {gp} Grand Prix in {year}."
    )
    
