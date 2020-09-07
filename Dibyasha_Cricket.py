
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


matches = pd.read_csv("F:\ContinousDataset.csv", index_col=0)
original = pd.read_csv("F:\originalDataset.csv", index_col=0)
Category = pd.read_csv("F:\CategoricalDataset.csv")
Labelled= pd.read_csv("F:\LabelledDataset.csv")


# In[3]:


matches.head()


# In[4]:


original.head()


# In[5]:


#Total Number of ODI Matches Played:
print(" Number of ODI Matches Played:" ,matches.shape[0])
#Teams Played:
print("\nTeams:\n", matches['Team 1'].unique())
#Number of Grounds for Matches:
print("\nNumber of Grounds Used:", matches['Ground'].nunique())
#Matches Played at Neutral Venue:
print("Number of Neutral Ground Matches:", matches.loc[matches.Venue_Team1 == 'Neutral'].shape[0])


# In[6]:


print('Matches Won By Home Country:\n',
     round(matches.loc[matches.Winner == matches.Host_Country].shape[0]/7494*100,2), '%')
print('\nNumber of times Home Team Batted First:\n',
     round(matches.loc[((matches['Venue_Team1'] == 'Home') & (matches['Innings_Team1'] == 'First')) |
                       ((matches['Venue_Team2'] == 'Home') & (matches['Innings_Team2'] == 'First'))].shape[0]/7494*100,2), 
      '%')


# In[7]:


print('Winning Team Based on Innings in %:\n', matches.Margin.value_counts()/7494*100)


# In[8]:


print('Most Matches won at Neutral Venues:', matches.loc[matches.Venue_Team1 == 'Neutral'].Winner.value_counts().idxmax())


# In[9]:


matches['Year'] = matches['Match Date'].str[-4:]
yearwise = matches['Year'].value_counts().sort_values( ascending= False ).reset_index()
yearwise.columns =['Year', 'Matches']
yearwise =yearwise.sort_values(by='Year', ascending=True)
sns.set_style('darkgrid')
plt.figure(figsize=(20,10))
plt.bar(yearwise['Year'], yearwise['Matches'], width=0.65, color='orange')
plt.xticks(rotation='vertical', size=15)
plt.yticks(size=15)
plt.xlabel('Year', size=20)
plt.ylabel('No. of Matches', size=20)
plt.title('Matches per year', size=25)
plt.show


# In[10]:


original['Winner'].value_counts().keys()[0] # Most number of wins by a team from 1971 to 2017


# In[11]:


original_india_australia = original[(original['Team 1'].str.contains('Australia') | original['Team 1'].str.contains('India'))  & (original['Team 2'].str.contains('India') | original['Team 2'].str.contains('Australia'))]


# In[12]:


original_india_australia['Winner'].value_counts()


# In[13]:


def homewin(row):
    if row['Winner'] == row['Host_Country']:
        return row['Winner']
def awaywin(row):
    if ((row['Winner'] == row['Team 1']) and (row['Venue_Team1'] != 'Home')) or ((row['Winner'] == row['Team 2']) 
                                                                                 and (row['Venue_Team2'] != 'Home')):
        return row['Winner']
def homematches(row):
    if (row['Host_Country'] == row['Team 1']):
        return row['Team 1']
    elif (row['Host_Country'] == row['Team 2']):
        return row['Team 2']
def wins(row):
    if (row['Team 1'] == row['Winner']):
        return row['Team 1']
    elif (row['Team 2'] == row['Winner']):
        return row['Team 2']
def dwin(row):
    if (row['Team 1'] == row['Winner']) and (row['Innings_Team1'] == 'First'):
        return row['Team 1']
    elif (row['Team 2'] == row['Winner']) and (row['Innings_Team2'] == 'First'):
        return row['Team 2']
def cwin(row):
    if (row['Team 1'] == row['Winner']) and (row['Innings_Team1'] == 'Second'):
        return row['Team 1']
    elif (row['Team 2'] == row['Winner']) and (row['Innings_Team2'] == 'Second'):
        return row['Team 2']


# In[14]:


matches['Home Win'] = matches.apply(homewin, axis='columns')
matches['Away Win'] = matches.apply(awaywin, axis='columns')
matches['Home Matches'] = matches.apply(homematches, axis = 'columns')
matches['Wins'] = matches.apply(wins, axis ='columns')
matches['Defend'] = matches.apply(dwin, axis ='columns')
matches['Chase'] = matches.apply(cwin, axis ='columns')


# In[15]:


Review = pd.concat([matches['Team 1'], matches["Team 2"]])
Review = Review.value_counts()/2
Review = Review.reset_index()
Review.columns=['Teams', 'Total Matches']
Review.set_index('Teams', inplace=True)


# In[16]:


hmatches = matches['Home Matches'].value_counts().reset_index().set_index('index')
awin = matches['Away Win'].value_counts().reset_index().set_index('index')
hwin = matches['Home Win'].value_counts().reset_index().set_index('index')
wins = matches['Wins'].value_counts().reset_index().set_index('index')
defended = matches['Defend'].value_counts().reset_index().set_index('index')
chased = matches['Chase'].value_counts().reset_index().set_index('index')


# In[17]:


Review['Home Matches'] = hmatches['Home Matches']/2
Review['Away Matches'] = Review['Total Matches'] - Review['Home Matches']
Review['Total Wins'] = wins['Wins']/2
Review['Home Win'] = hwin['Home Win']/2
Review['Away Win'] = awin['Away Win']/2
Review['Successful Defences'] = defended['Defend']/2
Review['Successful Chases'] = chased['Chase']/2


# In[18]:


sns.set_style('darkgrid')
plt.subplots(figsize=(20,10))
m1 = plt.bar( Review.index , Review['Total Matches'], width=0.8, color='blue', edgecolor='black')
m2 = plt.bar( Review.index , Review['Home Matches'], width=0.8, color='red', edgecolor='black')
plt.xticks(Review.index,Review.index, rotation='vertical', size=15)
plt.yticks(size=15)
plt.show
plt.title('Matches Played (Home & Away)', size=25)
plt.xlabel('Country', size=20)
plt.ylabel('No. of Matches', size=20)
plt.legend((m1[0], m2[0]) ,('Away', 'Home'), prop={"size" :20}, loc=1)


# In[19]:


sns.set_style('darkgrid')
plt.subplots(figsize=(20,10))
a1 = plt.bar( Review.index , Review['Total Wins'], width=0.8, color='lightgreen', edgecolor='black')
h1 = plt.bar( Review.index , Review['Home Win'], width=0.8, color='sandybrown', edgecolor='black')
plt.xticks(Review.index,Review.index, rotation='vertical', size=15)
plt.yticks(size=15)
plt.show
plt.title('Matches Won (Home & Away)', size=25)
plt.xlabel('Country', size=20)
plt.ylabel('No. of Matches', size=20)
plt.legend((a1[0], h1[0]) ,('Away Victory', 'Home Victory'), prop={"size" :20}, loc=1)
plt.show


# In[20]:


sns.set_style('darkgrid')
plt.figure(figsize=(20,10))
w1 = plt.bar( Review.index , Review['Successful Defences'], width=0.8, color='crimson', edgecolor='black')
w2 = plt.bar( Review.index , Review['Successful Chases'], width=0.8, bottom=Review['Successful Defences'], color='grey'
            ,edgecolor='black')
plt.xticks(Review.index,Review.index, rotation='vertical', size=15)
plt.yticks(size=15)
plt.show
plt.title('Matches Won\n(Innings)', size=25)
plt.xlabel('Country', size=20)
plt.ylabel('No. of Matches', size=20)
plt.legend((w1[0], w2[0]) ,('Defended Totals', 'Chased Totals'), prop={"size" :20}, loc=1)
plt.show


# In[21]:


host = matches['Host_Country'].value_counts()/2
host = host.reset_index()
host.columns =['Country', 'Matches Hosted']
plt.figure(figsize=(20,10))
plt.bar(host['Country'], host['Matches Hosted'], width=.75, color='coral', edgecolor='red')
plt.xticks(rotation='vertical', size=15)
plt.yticks(size=15)
plt.xlabel('Host Country', size=20)
plt.ylabel('No. of Matches', size=20)
plt.title('Matches Hosted by Country', size=25)
plt.show


# In[22]:


def ywin(country):
    data = matches.loc[matches.Winner == country].groupby('Year').apply(lambda p : p.Winner.value_counts()/2)
    return data
for team in matches['Winner'].unique():
    ydata = ywin(team)
    if team == 'Australia':
        yearwisewin = ydata
        yearwisewin
    elif team != 'Australia':
        yearwisewin = yearwisewin.join(ydata)
        yearwisewin


# In[23]:


sns.set_style('darkgrid')
plt.figure(figsize=(18,8))
sns.heatmap(data=yearwisewin.transpose(), annot=True, cmap='Blues', linewidth=.05)
plt.xlabel('Year', size=15)
plt.ylabel('Country', size=15)
plt.title('Victories each Year\n(Country)', size=20)
plt.show


# In[25]:


def cwin(country):
    played = original.loc[((original['Team 1'] == country) | (original['Team 2'] == country)) & 
                          (original['Winner'] == country)]
    wplayed = played.groupby(['Team 1', 'Team 2']).apply(lambda p : p.Winner.value_counts()).reset_index()
    w1played = wplayed.loc[wplayed['Team 1'] == country]
    w1played = w1played.rename(columns={'Team 2': 'Against','Team 1': 'Country' })
    w2played = wplayed.loc[wplayed['Team 2'] == country]
    w2played = w2played.rename(columns={'Team 1': 'Against','Team 2': 'Country' })
    finalwon = pd.concat([w1played, w2played]).groupby('Against').apply(lambda p : p[country].sum()).reset_index()
    finalwon = finalwon.set_index('Against')
    finalwon.columns = [country]
    
    nmatches = original.loc[(original['Team 1'] == country) | (original['Team 2'] == country)]
    mplayed = nmatches.groupby(['Team 1']).apply(lambda p : p['Team 2'].value_counts()).reset_index()
    m1played = mplayed.loc[mplayed['Team 1'] == country]
    m1played = m1played.rename(columns={'level_1': 'Against','Team 1': 'Country' })
    m2played = mplayed.loc[mplayed['level_1'] == country]
    m2played = m2played.rename(columns={'Team 1': 'Against','level_1': 'Country' })
    finalplayed = pd.concat([m1played, m2played]).groupby('Against').apply(lambda p : p['Team 2'].sum()).reset_index()
    finalplayed = finalplayed.set_index('Against')
    finalplayed.columns = [country]
    
    winpercent = finalwon/finalplayed
    return winpercent


teamnames = original['Winner'].unique()
teamnames = teamnames[(teamnames != 'tied') & (teamnames != 'no result') & (teamnames != 'ICC World XI')
                     & (teamnames != 'Asia XI') & (teamnames != 'Africa XI')]

for team in teamnames:
    cdata = cwin(team)
    if team == 'Australia':
        oppwisewin = cdata
        oppwisewin
    elif team != 'Australia':
        oppwisewin = oppwisewin.join(cdata)
        oppwisewin = oppwisewin.transpose()
        oppwisewin['Australia'] = 1 - oppwisewin.transpose().Australia
        oppwisewin = oppwisewin.transpose()
        oppwisewin


# In[26]:


sns.set_style('dark')
plt.figure(figsize=(18,8))
sns.heatmap(data=oppwisewin.transpose(), annot=True, cmap='YlOrBr')
plt.xlabel('Opponent', size=15)
plt.ylabel('Team', size=15)
plt.title('Teams Performance\n(Opponent)', size=20)
plt.show


# In[27]:


matches = pd.read_csv("F:\ContinousDataset.csv", index_col=0)


# In[28]:


labels = matches[['Winner']]
data = matches[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]


# In[29]:


data.head(5)


# In[30]:


data_new = pd.get_dummies(data)
data_new.head(5)


# In[31]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels) 


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_new, labels, test_size=0.1)


# In[33]:


from sklearn.ensemble import RandomForestClassifier as rfc
clf = rfc(n_estimators=100, max_depth=2, random_state=0)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
len(preds[preds == y_test])/len(preds)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier as knn
clf = knn(n_neighbors=10)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
len(preds[preds == y_test])/len(preds)


# In[35]:


data.head(5)


# # Real training team by team

# In[36]:


matches = pd.read_csv("F:\ContinousDataset.csv", index_col=0)


# In[37]:


data = matches[((matches['Team 1'] == 'Australia') | (matches['Team 1'] == 'West Indies')) & ((matches['Team 2'] == 'Australia') | (matches['Team 2'] == 'West Indies'))]


# In[38]:


labels = data[['Winner']]
data = data[['Team 1', 'Team 2', 'Ground', 'Host_Country', 'Venue_Team1', 'Venue_Team2', 'Innings_Team1', 'Innings_Team2']]


# In[39]:


data_hot = pd.get_dummies(data) 


# In[40]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_hot, labels, test_size=0.1)


# In[41]:


from sklearn.ensemble import RandomForestClassifier as rfc
clf = rfc(n_estimators=5, max_depth=2, random_state=0)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
preds = preds.reshape(preds.shape[0],1)
len(preds[preds == y_test])/len(preds)


# In[42]:


from sklearn.tree import DecisionTreeClassifier as DTC
clf = DTC(random_state=0)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
preds = preds.reshape(preds.shape[0],1)
len(preds[preds == y_test])/len(preds)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier as knn
clf = knn(n_neighbors=3)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
preds = preds.reshape(preds.shape[0],1)
len(preds[preds == y_test])/len(preds)

