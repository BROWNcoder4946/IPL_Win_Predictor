import streamlit as st
import pickle
import pandas as pd



teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb')) # Give full path
st.title('IPL Win Predictor')


col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select The batting Team',teams)
with col2:
    bowling_team=st.selectbox('Select the bowling team',teams)
if batting_team == bowling_team:
    st.warning("Teams should not be the same. Please enter different team.")
    butcon=1
else: 
    butcon=0
selected_city=st.selectbox('Select host City',cities)

target=st.number_input('target',value=1, step=1, min_value=1)

col3,col4,col5=st.columns(3)

with col3:
    score=st.number_input('Score',value=0, step=1,max_value=target-1, min_value=0)

with col4:
    overs=st.number_input('Overs Completed',value=1, step=1, max_value=19, min_value=1)

with col5:
    wickets=st.number_input('Wickets Out',value=0, step=1, max_value=9, min_value=0)

if st.button('predict probability',disabled=butcon) : 
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets_left=10-wickets
    crr=score/overs
    rrr=runs_left/(20-overs)
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})


    result=pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")