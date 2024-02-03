# %%
import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# %%
match = pd.read_csv('matches.csv') # Give full path
delivery = pd.read_csv('deliveries.csv') # Give full path

# %%
match.head()

# %%
match.shape # To find dimensions

# %%
delivery.head()

# %%
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df #Creating total score from given attributes 

# %%
total_score_df = total_score_df[total_score_df['inning'] == 1] #Finding target score only

# %%
total_score_df

# %%
#Appending target column to match_df
match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id') 

# %%
match_df

# %%
match_df['team1'].unique()

# %%
teams = [ # teams presently playing
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

# %%
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

# %%
# Removing non-playing teams
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# %%
match_df.shape

# %%
# Not worth it
match_df = match_df[match_df['dl_applied'] == 0]

# %%
match_df = match_df[['match_id','city','winner','total_runs']]

# %%
delivery_df = match_df.merge(delivery,on='match_id')

# %%
# Keeping only 2nd innings
delivery_df = delivery_df[delivery_df['inning'] == 2]

# %%
delivery_df

# %%
# Calculating current score per ball
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

# %%
# Calculating runs left per each ball
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

# %%
# Calculating ballls left per each ball
delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])

# %%
delivery_df

# %%
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0") # Where na , put 0
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1") # Otherwise put 1
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets_left'] = 10 - wickets
delivery_df.head(20)

# %%
delivery_df.head()

# %%
# crr = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])

# %%
delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']

# %%
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

# %%
delivery_df['result'] = delivery_df.apply(result,axis=1)

# %%
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','crr','rrr','result']]

# %%
# Shuffling data for more accurate results
final_df = final_df.sample(final_df.shape[0])

# %%
final_df.sample()

# %%
# Removing rows with NA
final_df.dropna(inplace=True)

# %%
# Removing balls left with 0 (Error prone)
final_df = final_df[final_df['balls_left'] != 0]

# %%
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# %%
X_train

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    # Basically Dummy encoding
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough') # Do not disturb other columns

# %%
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# %%
#The idea behind the pipeline is to concatenate these steps into a single object (pipe). 
#This can be especially useful when you want to perform a series of transformations on your 
#data before fitting a model, and store results in a variable
pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
    #('step2',RandomForestClassifier())
])

# %%
pipe.fit(X_train,y_train)
#To view coefficients of trained attributess
# logreg_model = pipe.named_steps['step2']
# coefficients = logreg_model.coef_
# intercept = logreg_model.intercept_

# %%
# feature_names = pipe.named_steps['step1'].get_feature_names_out(input_features=X_train.columns)  # Assuming 'step1' is the name you provided for the ColumnTransformer step
# for feature, coef in zip(feature_names, coefficients[0]):
#     print(f'{feature}: {coef}')

# %%
# intercept

# %%
y_pred = pipe.predict(X_test)

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# %%
# pipe.predict_proba(X_test)[10]

# %%
import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))


