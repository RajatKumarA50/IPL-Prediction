# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 12:30:05 2020

@author: Rajat
"""


# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt

# Reading the dataset
matches=pd.read_csv('res.csv')
matches.info()
matches[pd.isnull(matches['winner'])]
matches['winner'].fillna('Draw', inplace=True)

matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Capitals','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','CSK','RR','DC','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DC':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'team2': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DC':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DC':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12},
          'winner': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':5,'DC':6,'GL':7,'KXIP':8,'SRH':9,'RPS':10,'KTK':11,'PW':12,'Draw':13}}
matches.replace(encode, inplace=True)
matches.head(2)

#Find cities which are null
matches[pd.isnull(matches['city'])]

#remove any null values, winner has hence fill the null value in winner as draw
#City is also null
matches['city'].fillna('Dubai',inplace=True)
matches.describe()

dicVal = encode['winner']
print(dicVal['MI']) #key value
print(list(dicVal.keys())[list(dicVal.values()).index(1)]) #find key by value search 

matches = matches[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
matches.head(2)

df = pd.DataFrame(matches)
df.describe()

temp1=df['toss_winner'].value_counts(sort=True)
temp2=df['winner'].value_counts(sort=True)
#Mumbai won most toss and also most matches
print('No of toss winners by each team')
for idx, val in temp1.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
print('No of match winners by each team')
for idx, val in temp2.iteritems():
   print('{} -> {}'.format(list(dicVal.keys())[list(dicVal.values()).index(idx)],val))
   
#shows that Mumbai won most matches followed by Chennai
df['winner'].hist(bins=40)   

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Team')
ax1.set_ylabel('Count of toss wins')
ax1.set_title("toss winners")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Team')
ax2.set_ylabel('count of matches won')
ax2.set_title("Match winners")

df.apply(lambda x: sum(x.isnull()),axis=0) 
    #find the null values in every column
 
 #Find cities which are null
df[pd.isnull(df['city'])]

#building predictive model
from sklearn.preprocessing import LabelEncoder
var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes
pe: object

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner']
model = LogisticRegression()
from sklearn import classification_model 
classification_model(model, df,predictor_var,outcome_var)

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
  kf = KFold(data.shape[0], n_folds=7)
  error = []
  for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
  print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))
  model.fit(data[predictors],data[outcome])   

from sklearn.ensemble import RandomForestRegressor
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner']
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classification_model(model, df,predictor_var,outcome_var)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestClassifier(n_estimators=100)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)

df.head(7)

#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
team1='SRH'
team2='KXIP'
toss_winner='KXIP'
input=[dicVal[team1],dicVal[team2],'14',dicVal[toss_winner],'2','1']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)]) #find key by value search output

#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
team1='SRH'
team2='CSK'
toss_winner='CSK'
input=[dicVal[team1],dicVal[team2],'23',dicVal[toss_winner],'14','0']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)]) #find key by value search output

#feature importances: If we ignore teams, Venue seems to be one of important factors in determining winners 
#followed by toss winning, city
#we notice that team1 and team2 account for highest value for reason that it is either if these value 
#with toss winner going to be winner. So we could ignore team2 and team1
imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(imp_input)

#okay from the above prediction on features, we notice toss winner has least chances of winning matches
#but does the current stats shows the same result
#df.count --> 577 rows
#Previously toss_winners were about 50.4%, with 2017 IPL season, it has reached 56.7%. As data matures, so does
# the changes in the predictions
mlt.style.use('fivethirtyeight')
df_fil=df[df['toss_winner']==df['winner']]
slices=[len(df_fil),(577-len(df_fil))]
mlt.pie(slices,labels=['Toss & win','Toss & lose'],startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%',colors=['r','g'])
fig = mlt.gcf()
fig.set_size_inches(6,6)
mlt.show()
# Toss winning does not gaurantee a match win from analysis of current stats and thus 
#prediction feature gives less weightage to that 

#top 2 team analysis based on number of matches won against each other and how venue affects them?
#Previously we noticed that CSK won 79, RCB won 70 matches
#now let us compare venue against a match between CSK and RCB
#we find that CSK has won most matches against RCB in MA Chidambaram Stadium, Chepauk, Chennai
#RCB has not won any match with CSK in stadiums St George's Park and Wankhede Stadium, but won matches
#with CSK in Kingsmead, New Wanderers Stadium.
#It does prove that chances of CSK winning is more in Chepauk stadium when played against RCB.
# Proves venue is important feature in predictability
import seaborn as sns
team1=dicVal['MI']
team2=dicVal['CSK']
mtemp=matches[((matches['team1']==team1)|(matches['team2']==team1))&((matches['team1']==team2)|(matches['team2']==team2))]
sns.countplot(x='venue', hue='winner',data=mtemp,palette='Set2')
mlt.xticks(rotation='vertical')
leg = mlt.legend( loc = 'upper right')
fig=mlt.gcf()
fig.set_size_inches(10,6)
mlt.show()

#use- le.classes_[18] to get stadium details
le.classes_[15]

