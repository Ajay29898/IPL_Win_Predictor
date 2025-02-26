import pickle
import pandas as pd
import streamlit as st

# Set the page title and icon
st.set_page_config(page_title="IPL Match Win Predictor", page_icon="🏆")

st.markdown("<h1 style='text-align: center; font-weight: bold;'>IPL Match Win Predictor 🏆</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1531415074968-036ba1b575da?q=80&w=2067&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

col1, col2 = st.columns([1, 1])

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams), index=None, placeholder="Select Batting Team")
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams), index=None, placeholder="Select Bowling Team")

col3, col4 = st.columns([1, 1])

with col3:
    selected_city = st.selectbox('Select host city', sorted(cities), index=None, placeholder="Select Host City")

with col4:
    target = st.number_input('Target', min_value=1, max_value=300, )

col5, col6, col7 = st.columns([3, 3, 3])

with col5:
    score = st.number_input('Score', min_value=0, max_value=target - 1)
with col6:
    overs = st.number_input('Overs completed', min_value=0, max_value=20)
with col7:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

if st.button('Predict Probability 🔃'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets_left],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    if batting_team != bowling_team:
        st.markdown(f"<h2 style='font-weight: bold;'>{batting_team} - {round(win * 100)}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='font-weight: bold;'>{bowling_team} - {round(loss * 100)}%</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please select different batting and bowling teams.")