import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
@st.cache_data
def load_model():
    with open('models/rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model()

# Streamlit UI
st.title(" Fake Social Media Account Detection")
st.write("Enter the details below to check if an account is **real** or **fake**.")

# User inputs
abuse_reports = st.number_input('Number of Abuse Reports', min_value=0)
rejected_friend_requests = st.number_input('Number of Rejected Friend Requests', min_value=0)
unaccepted_friend_requests = st.number_input('Number of Friend Requests Not Accepted', min_value=0)
friends_count = st.number_input('Number of Friends', min_value=0)
followers_count = st.number_input('Number of Followers', min_value=0)
likes_to_unknown_accounts = st.number_input('Number of Likes to Unknown Accounts', min_value=0)
comments_per_day = st.number_input('Number of Comments Per Day', min_value=0)

# Prepare input data
input_data = pd.DataFrame({
    'AbuseReports': [abuse_reports],
    'RejectedFriendRequests': [rejected_friend_requests],
    'UnacceptedFriendRequests': [unaccepted_friend_requests],
    'FriendsCount': [friends_count],
    'FollowersCount': [followers_count],
    'LikesToUnknownAccounts': [likes_to_unknown_accounts],
    'CommentsPerDay': [comments_per_day]
})

# Predict button
if st.button(" Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    result = " **Fake Account** " if prediction[0] == 1 else " **Real Account** "
    st.write(f"### {result}")
