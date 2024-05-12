#!/usr/bin/env python
# coding: utf-8

# In[222]:


import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from textblob import TextBlob
import json
import json
import pandas as pd
from nltk.corpus import state_union

import pandas as pd
from textblob import TextBlob
#import neuralcoref
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Chart of alot of Entites, not final product, just checking (charts take time to be made)

# In[234]:


letsgo = pd.read_csv("Data_Processing/entities_counted_Partied.csv")
letsgo


# In[235]:


# Assuming df is your DataFrame

# Initialize a Min-Max Scaler
scaler = MinMaxScaler()
letgos = letsgo.drop(columns=letsgo.columns[0])
letgos = letgos.drop(columns=letgos.columns[0])

# Scale the data
letgos_scaled = pd.DataFrame(scaler.fit_transform(letgos), columns=letgos.columns, index=letsgo["_Year_"])

# Create the heatmap
plt.figure(figsize=(50, 100))
sns.heatmap(letgos_scaled.T, annot=False, cmap='Blues')
plt.show()



# # Obviously the above Data Visulization is just testign around with the data

# ## Chart 1

# In[236]:


data = [
    "Europe: 1037", "Germany: 386", "Hitler: 56", "Jews: 34", 
    "World War II: 157", "Ukraine: 137", "Turkey: 134"
]

# Splitting each string at ':' and taking the first part
cleaned_list_Europe = [(item.split(':')[0] )for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_Europe

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20)
plt.ylabel('Sum', fontsize=20)
plt.title('World War Entites Sum Non Democrat' , fontsize=20)
plt.xticks(rotation=45, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)

# Add a caption
plt.figtext(0.5, -0.4, "The bar chart displays the frequency of specific entities mentioned in all Republicn President Speeches since US presidency specific to World Wars.", wrap=True, horizontalalignment='center', fontsize=13)

plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=45, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('World War Entites Sum Non Republican')

# Add a caption
plt.figtext(0.5, -0.4, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to World Wars.", wrap=True, horizontalalignment='center', fontsize=13)

plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_Europe]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('World War Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Democrat, including whips and non denomination parties, President Speeches since US presidency specific to World Wars in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_Europe]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)


# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('World War Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to World Wars in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)


#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_Europe]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)


# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('World War Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to World Wars in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,7))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('World War Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to World Wars in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 7))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('World War Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to World Wars in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 2

# In[237]:


data = [
    "Cuba: 618", "Panama: 261", "Mexico: 978", "Nicaragua: 224",
    "Cuban: 174", "the Panama Canal: 73", "Porto Rico: 72"
]

# Splitting each string at ':' and taking the first part
cleaned_list_South_American = [(item.split(':')[0] )for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_South_American

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entities', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('South American Entites Sum Non Democrat', fontsize =20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to South American Involvement.", wrap=True, horizontalalignment='center', fontsize=13)

plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entities', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation= 75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('South American Entites Sum Non Republican', fontsize = 20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to South American Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_South_American]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('South American Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to South American Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_South_American]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('South American Speech Entities Over The Years Non Republican\n', fontsize=25)
# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to South American Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_South_American]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('South American Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to South American Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,7))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('South American Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to South American Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 7))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('South American Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to South American Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 3

# In[238]:


data = [
    "the Soviet Union: 613", "Russia: 467", "Soviet: 450",
    "Communist: 354", "Soviets: 177", "Communists: 175"
]


# Splitting each string at ':' and taking the first part
# Splitting each string at ':' and taking the first part
cleaned_list_russian = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_russian

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Russian Entites Sum Non Democrat', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.55, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to Russian Involvements.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Russian Entites Sum Non Republican', fontsize= 20)

# Add a caption
plt.figtext(0.5, -0.55, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to Russian Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_russian]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Russian Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to Cold War Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_russian]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Russian Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to Cold War Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_russian]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Russian Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to Russian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,7))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Russian Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to Russian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 7))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Russian Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.1, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to Russian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)


#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 4

# In[239]:


data = [
    "Iraq: 590", "the Middle East: 363", "Afghanistan: 458", "Iran: 282", 
    "Iraqi: 191", "Taliban: 118", "Afghan: 112", "al Qaeda: 106", 
    "Arab: 99", "Saddam Hussein: 75", "Saddam Hussein's: 13", 
    "Iraqis: 75", "Somalia: 73", "Muslim: 48", "Al Qaeda: 45", 
    "Osama bin Laden: 30", "Afghans: 30", "Assad: 30"
]



# Splitting each string at ':' and taking the first part
# Splitting each string at ':' and taking the first part
cleaned_list_Middle_east = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_Middle_east

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Middle East Entites Sum Non Democrat', fontsize= 20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to Middle Eastern Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Middle East Entites Sum Non Republican', fontsize= 20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to Middle Eastern Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_Middle_east]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Middle East Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to Middle Eastern Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_Middle_east]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Middle East Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to Middle Eastern Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_Middle_east]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Middle East Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to Middle East Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,10))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Middle East Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to Middle East Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Middle East Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to Middle East Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 5

# In[240]:


data = [
    "Vietnam: 473", "Japan: 387", "Viet-Nam: 145", "South Vietnam: 165",
    "Korea: 174", "Philippines: 160", "Japanese: 157",
    "South Vietnamese: 112", "South Viet-Nam: 111", "North Vietnam: 111",
    "China: 841", "Chinese: 361"
]



# Splitting each string at ':' and taking the first part
# Splitting each string at ':' and taking the first part
cleaned_list_Asia = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_Asia

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Asian Entites Sum Non Democrat', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to Asian Involvement.", wrap=True, horizontalalignment='center', fontsize=13)

plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Asian Entites Sum Non Republican', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to Asian Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_Asia]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Asia Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to Asian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_Asia]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Asia Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to Asian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_Asia]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Asia Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to Asian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Asia Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to Asian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Asia Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to Asian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 6

# In[241]:


data = [
    "Israel: 332", "Palestinians: 70", "Palestinian: 68", 
    "Israelis: 62", "Palestine: 26", "Israeli: 58", "Jews: 34"
]


# Splitting each string at ':' and taking the first part
cleaned_list_I_P = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_I_P

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=45, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('I/P Entites Sum Non Democrat', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.4, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to Isreal and Palestinian Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=45, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('I/P Entites Sum Non Republican', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.4, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to Israel and Palestinian Involvement.", wrap=True, horizontalalignment='center', fontsize=13)

plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_I_P]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('I/P Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to Israel and Palestinian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_I_P]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('I/P Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to Israel and Palestinian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_I_P]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('I/P Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to Israel and Palestinian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,7))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('I/P Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to Israel and Palestinian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 7))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('I/P Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to Israel and Palestinian Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 7

# In[242]:


data = [
    "Negro: 84", "the Civil War: 56", "African American: 34", 
    "Negroes: 25", "Africans: 22", "African-American: 15", 
    "African-Americans: 15", "Native American: 15"
]


# Splitting each string at ':' and taking the first part
cleaned_list_Minority = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_Minority

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Minority Entites Sum Non Democrat', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency specific to Minority Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=75, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Minority Entites Sum Non Republican', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.6, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency specific to Minority Involvement.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_Minority]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Minority Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency specific to Minority Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_Minority]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Minority Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Republican, including whips and non denomination parties, President Speeches since US presidency specific to Minority Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_Minority]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 7))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Minority Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency specific to Minority Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25,7))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Minority Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency specific to Minority Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 7))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Minority Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency specific to Minority Involvement in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# ## Chart 8

# In[243]:


data = [
    "the United Nations: 407", "Medicare: 252", "Ford: 190", "Kennedy: 276",
    "Trump: 91", "Truman: 74", "Lincoln: 73", "Martin Luther King: 31",
    "Bush: 163", "Reagan: 149", "Nixon: 147", "Pakistan: 81",
    "Watergate: 73", "the Civil War: 56", "COVID-19: 62", "Great Britain: 679",
    "British: 603"
]



# Splitting each string at ':' and taking the first part
cleaned_list_assorted = [item.split(':')[0] for item in data]


##
##
##
non_democrat_df = letsgo[letsgo['_Party_'] != 'Democrat']
non_Republican_df = letsgo[letsgo['_Party_'] != 'Republican']

import matplotlib.pyplot as plt

# Assuming non_democrat_df is your DataFrame
subs = cleaned_list_assorted

# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_democrat_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Red')
plt.xlabel('Entites', fontsize=20, labelpad=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Other Entites Sum Non Democrat', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.65, "The bar chart displays the frequency of specific entities mentioned in all Republican President Speeches since US presidency non-specific to any event or topic but that seemed of interest.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()


# Calculate the sum of each column
sums = []
for column in subs:
    sums.append(non_Republican_df[column].sum())

# Create a bar chart using the sums
plt.bar(subs, sums, color='Blue')
plt.xlabel('Entites', fontsize=20, labelpad=20 )
plt.ylabel('Sum', fontsize=20)
plt.xticks(rotation=90, fontsize=15)  # Rotate the x labels for better readability
plt.yticks( fontsize=15)
plt.title('Other Entites Sum Non Republican', fontsize=20)

# Add a caption
plt.figtext(0.5, -0.65, "The bar chart displays the frequency of specific entities mentioned in all Democrat President Speeches since US presidency non-specific to any event or topic but that seemed of interest.", wrap=True, horizontalalignment='center', fontsize=13)


plt.show()




def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

subs=non_democrat_df[cleaned_list_assorted]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Reds')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Other Speech Entities Over The Years Non Democrat\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency non-specific to any event or topic but that seemed of interest, in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()






subs=non_Republican_df[cleaned_list_assorted]

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()


result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)

# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Blues')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Other Speech Entities Over The Years Non Republican\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all Non-Democratic, including whips and non denomination parties, President Speeches since US presidency non-specific to any event or topic but that seemed of interest, in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

##
##
##


subs=letsgo[cleaned_list_assorted]
def map_to_interval(year):
    return f"{(year // 10) * 10}-{(year // 10) * 10 + 9}"

# Assign years to intervals
subs['Interval'] = letsgo["_Year_"].astype(int).map(map_to_interval)

# Group by interval and sum
result_df = subs.groupby('Interval').sum()

result_df.index= result_df.index.astype(str).str[:4]
result_df.rename_axis("year", inplace=True)



# Set the size of the heatmap
plt.figure(figsize=(25, 10))  # You can adjust the size as needed

# Create the heatmap
sns.heatmap(result_df.T, annot=False, cmap='Purples')  # 'annot=False' hides the data values, 'cmap' sets the color map

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Other Speech Entities Over The Years\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the frequency of specific entities mentioned in all President Speeches since US presidency non-specific to any event or topic but that seemed of interest, in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()

# Assuming df is your DataFrame



# Initialize a Min-Max Scaler
scaler = MinMaxScaler()

# Scale the data
subs_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(subs_scaled.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Other Speech Entities Over The Years MinMax Scaled\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in all President Speeches since US presidency non-specific to any event or topic but that seemed of interest, in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()



# Initialize a Standard Scaler
scaler = StandardScaler()

# Scale the data
letsgo_standardized = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)

# Create the heatmap
plt.figure(figsize=(25, 10))
sns.heatmap(letsgo_standardized.T, annot=False, cmap='Purples')

plt.xlabel('Year', fontsize = 20)
plt.yticks(fontsize = 20, rotation = 360)
plt.xticks(fontsize = 17)
plt.title('Other Speech Entities Over The Years Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.05, "The heatmap displays the Standardized frequency of specific entities mentioned in all President Speeches since US presidency non-specific to any event or topic but that seemed of interest, in a temporal dimension cut up by decades.", wrap=True, horizontalalignment='center', fontsize=20)

#plt.savefig('Speech_heatmap.png');
plt.show()


# # Case Studies of Entites mentioned compared from President to President

# ## Case Study 1 World War II Entites

# In[244]:


samp_sorted= pd.read_csv('Data_Processing/outputfinal_filename.csv')

startyear=1939
endyear=1945


startDate=19390901 # Invasion of Poland, attack on Poland by Nazi Germany that marked the start of World War II. The invasion lasted from September 1 to October 5, 1939. As dawn broke on September 1, 1939, German forces launched a surprise attack on Poland.
endDate=19450814 # US President Harry S. Truman announced Japan's surrender and the end of World War II

gents = [
    "Europe", "Germany", "Hitler", "Jews", 
    "World War II",  'Navy', 'Army', 'Poland', 'Japan', 'Japanese'
]










bummy = samp_sorted.loc[(samp_sorted['Year'] >= startyear) & (samp_sorted['Year'] <= endyear)]
bummy.reset_index(inplace=True)
bummy.drop('index', axis=1, inplace=True)



df1 = pd.DataFrame(columns=['doc_name', 'date', 'transcript', 'president', 'title', 'Year'])


for i in range(len(bummy)):
    date_string = bummy.loc[bummy.index[i], 'date']  # Accessing the 'date' column for each row
    cleaned_string = date_string.replace('-', '')
    # Convert string to number
    result = int(cleaned_string)
    
    if startDate <= result <= endDate:
        
        # Assuming df1 is the DataFrame you want to add a row to, and df2 is the DataFrame from which you want to add the row
        row_to_add = bummy.loc[i]  # Select the row you want to add from df2

        df1.loc[len(df1)] = row_to_add.values
        
        
        
        
# Ensure the 'date' column is in datetime format
df1['date'] = pd.to_datetime(df1['date'])


# Create a DataFrame with each president's first and last speech date
president_dates = df1.groupby('president')['date'].agg(['min', 'max']).reset_index()
president_dates.columns = ['president', 'first_speech', 'last_speech']

# Print the new DataFrame
print(president_dates)





# Create an empty DataFrame to store the counts
fumpy = pd.DataFrame(columns=['president'] + gents)

# Iterate over each president in df1
for president in df1['president'].unique():
    # Initialize a dictionary to store the counts for each president
    president_counts = {'president': president}
    # Filter df1 for speeches by the current president
    president_df = df1[df1['president'] == president]
    # Iterate over each word in gents
    for word in gents:
        # Count the occurrences of the word in the current president's speeches
        word_count = sum(president_df['transcript'].str.count(word))
        # Add the count to the dictionary
        president_counts[word] = word_count
    # Append the counts for the current president to fumpy
    fumpy = fumpy.append(president_counts, ignore_index=True)
    
    
# Set the 'president' column as the index
fumpy.set_index('president', inplace=True)

# Calculate the sum of each column and sort the columns by the sum in descending order
fumpy = fumpy.loc[:, fumpy.sum().sort_values(ascending=False).index]


# Print the DataFrame fumpy
print(fumpy)


# In[245]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

# Fill missing values with zeros
fumpy_filled = fumpy.fillna(0)

# Convert fumpy to numeric type
fumpy_numeric = fumpy_filled.apply(pd.to_numeric)





# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_transposed, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label == 'Franklin D. Roosevelt' else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 10.5  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx + 0.5, y_pos_below, label, ha='center', va='center', fontsize=20, color=color)
    start_date, end_date = date_mapping[label]
    plt.text(idx + 0.5, y_pos_below + 0.8, f'{start_date} - {end_date}', ha='center', va='center', fontsize=15, color='black')
    

legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility

plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=65)
plt.ylabel('Entities', fontsize=20)
plt.title(f'World War II Entity Frequency by Presidents \n {startyear} - {endyear} \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.2, "The heatmap displays the frequency of specific entities mentioned in President Speeches during the signifiacnt years of World War II, with annotations indicating the start and end dates of of presidents in office during that time period, which started with Invasion of Poland by Nazi Germany that marked the start of World War II and ended with the US President Harry S. Truman announcement of Japan's surrender. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()






# Initialize a StandardScaler
scaler = StandardScaler()

# Scale the data
fumpy_standardized = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_standardized, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label == 'Franklin D. Roosevelt' else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 10.5  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx + 0.5, y_pos_below, label, ha='center', va='center', fontsize=20, color=color)
    start_date, end_date = date_mapping[label]
    plt.text(idx + 0.5, y_pos_below + 0.8, f'{start_date} - {end_date}', ha='center', va='center', fontsize=15, color='black')

    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=65)
plt.ylabel('Entities', fontsize=20)
plt.title(f'World War II Entity Frequency by Presidents \n {startyear} - {endyear} Standardized \n', fontsize=25)


# Add a caption
plt.figtext(0.4, -0.2, "The heatmap displays the Standardized frequency of specific entities mentioned in President Speeches during the signifiacnt years of World War II, with annotations indicating the start and end dates of of presidents in office during that time period, which started with Invasion of Poland by Nazi Germany that marked the start of World War II and ended with the US President Harry S. Truman announcement of Japan's surrender. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()








# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# Scale the data
fumpy_scaled = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_scaled, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label == 'Franklin D. Roosevelt' else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 10.5  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx + 0.5, y_pos_below, label, ha='center', va='center', fontsize=20, color=color)
    start_date, end_date = date_mapping[label]
    plt.text(idx + 0.5, y_pos_below + 0.8, f'{start_date} - {end_date}', ha='center', va='center', fontsize=15, color='black')


legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility
    
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=65)
plt.ylabel('Entities', fontsize=20)
plt.title(f'World War II Entity Frequency by Presidents \n {startyear} - {endyear} MinMax Scaled \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.2, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in President Speeches during the signifiacnt years of World War II, with annotations indicating the start and end dates of of presidents in office during that time period, which started with Invasion of Poland by Nazi Germany that marked the start of World War II and ended with the US President Harry S. Truman announcement of Japan's surrender. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()


# ## Case Study 2 Cold War Entites

# In[246]:


samp_sorted= pd.read_csv('Data_Processing/outputfinal_filename.csv')

startyear=1947
endyear=1991


startDate=19470312 # The Cold War began with the announcement of the Truman Doctrine in 1947, started a gradual winding down with the Sino-Soviet split between the Soviets and the People's Republic of China in 1961, and ended with the collapse of the Soviet Union in 1991.
endDate=19911225 # Gorbachev resigned on 25 December 1991 and what was left of the Soviet parliament voted to end itself. the Soviet Union itself dissolved into its component republics. Sign of the end of the cold war.

gents = [
    "the Soviet Union", "Russia", "Soviet",
    "Communist", "Soviets", "Communists", 'Army', 'Navy'
]










bummy = samp_sorted.loc[(samp_sorted['Year'] >= startyear) & (samp_sorted['Year'] <= endyear)]
bummy.reset_index(inplace=True)
bummy.drop('index', axis=1, inplace=True)




df1 = pd.DataFrame(columns=['doc_name', 'date', 'transcript', 'president', 'title', 'Year'])

prev=startDate

for i in range(len(bummy)):
    date_string = bummy.loc[bummy.index[i], 'date']  # Accessing the 'date' column for each row
    cleaned_string = date_string.replace('-', '')
    
    # Convert string to number
    if 'T' in cleaned_string:
        cleaned_string=cleaned_string.split('T')[0]
    result = int(cleaned_string)
    if result<9999:
        
        original_str = str(prev)
        bluuuh = str(result)
    
        # Cut off the last four characters
        last_four = original_str[-4:]

        # Combine the remaining string with the new string
        res = bluuuh + last_four
        result =int(res)
        

    prev=result
    
    fumm=str(result)
    summmy=fumm[:4] + '-' + fumm[4:6] + '-' + fumm[6:]
    
    
    bummy.loc[bummy.index[i], 'date'] = summmy
    
    
    if startDate <= result <= endDate:
        
        # Assuming df1 is the DataFrame you want to add a row to, and df2 is the DataFrame from which you want to add the row
        row_to_add = bummy.loc[i]  # Select the row you want to add from df2

        df1.loc[len(df1)] = row_to_add.values
        
        
        

# Ensure the 'date' column is in datetime format
#df1['date'] = pd.to_datetime(df1['date'])

# Ensure the 'date' column is in datetime format
df1['date'] = pd.to_datetime(df1['date'])






# Create a DataFrame with each president's first and last speech date
president_dates = df1.groupby('president')['date'].agg(['min', 'max']).reset_index()

president_dates.columns = ['president', 'first_speech', 'last_speech']



# Print the new DataFrame
print(president_dates)





# Create an empty DataFrame to store the counts
fumpy = pd.DataFrame(columns=['president'] + gents)

# Iterate over each president in df1
for president in df1['president'].unique():
    # Initialize a dictionary to store the counts for each president
    president_counts = {'president': president}
    # Filter df1 for speeches by the current president
    president_df = df1[df1['president'] == president]
    # Iterate over each word in gents
    for word in gents:
        # Count the occurrences of the word in the current president's speeches
        word_count = sum(president_df['transcript'].str.count(word))
        # Add the count to the dictionary
        president_counts[word] = word_count
    # Append the counts for the current president to fumpy
    fumpy = fumpy.append(president_counts, ignore_index=True)
    
    
# Set the 'president' column as the index
fumpy.set_index('president', inplace=True)

# Calculate the sum of each column and sort the columns by the sum in descending order
fumpy = fumpy.loc[:, fumpy.sum().sort_values(ascending=False).index]


republican_presidents_df = pd.DataFrame({
    "Name": [
        "Dwight D. Eisenhower",
        "Richard M. Nixon",
        "Gerald Ford",
        "Ronald Reagan",
        "George H. W. Bush"
    ]
})


# Print the DataFrame fumpy
print(fumpy)


# In[247]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Fill missing values with zeros
fumpy_filled = fumpy.fillna(0)

# Convert fumpy to numeric type
fumpy_numeric = fumpy_filled.apply(pd.to_numeric)





# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_transposed, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 9  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx , y_pos_below, label, ha='center', va='center', fontsize=15, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+ .2, y_pos_below + .2, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)
    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility

plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=130)
plt.ylabel('Entities', fontsize=20)
plt.title(f'Cold War Entity Frequency by Presidents \n {startyear} - {endyear} \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.3, "The heatmap displays the frequency of specific entities mentioned in President Speeches during the signifiacnt years of Cold War, with annotations indicating the start and end dates of of presidents in office during that time period starting with the announcement of the Truman Doctrine in 1947 and ends with Gorbachev resigning and what was left of the Soviet parliament voted to end itself. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)


plt.show()







# Initialize a StandardScaler
scaler = StandardScaler()

# Scale the data
fumpy_standardized = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_standardized, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 9  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx , y_pos_below, label, ha='center', va='center', fontsize=15, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+ .2, y_pos_below + .2, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)

    
    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=130)
plt.ylabel('Entities', fontsize=20)
plt.title(f'Cold War Entity Frequency by Presidents \n {startyear} - {endyear} Standardized \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.3, "The heatmap displays the Standardized frequency of specific entities mentioned in President Speeches during the signifiacnt years of Cold War, with annotations indicating the start and end dates of of presidents in office during that time period starting with the announcement of the Truman Doctrine in 1947 and ends with Gorbachev resigning and what was left of the Soviet parliament voted to end itself. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)


plt.show()








# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# Scale the data
fumpy_scaled = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(15, 7))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_scaled, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 9  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx , y_pos_below, label, ha='center', va='center', fontsize=15, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+ .2, y_pos_below + .2, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)

    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    

plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=130)
plt.ylabel('Entities', fontsize=20)
plt.title(f'Cold War Entity Frequency by Presidents \n {startyear} - {endyear} MinMax Scaled \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.3, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in President Speeches during the signifiacnt years of Cold War, with annotations indicating the start and end dates of of presidents in office during that time period starting with the announcement of the Truman Doctrine in 1947 and ends with Gorbachev resigning and what was left of the Soviet parliament voted to end itself. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)


plt.show()


# In[ ]:





# In[ ]:





# ## Case Study 3 Middle Eastern Entities

# In[248]:


samp_sorted= pd.read_csv('Data_Processing/outputfinal_filename.csv')

startyear=1949
endyear=2023


startDate=19490816 #1949, August 16. Navy establishes Middle East Force. 
endDate=20230222 # Now, ealriest speach in data

gents = [
    "Iraq", "the Middle East", "Afghanistan", "Iran", 
    "Iraqi", "Taliban", "Afghan", "al Qaeda", 
    "Arab", "Saddam Hussein", "Saddam Hussein's", 
    "Iraqis", "Somalia", "Muslim", "Al Qaeda", 
    "Osama bin Laden", "Afghans", "Assad", "Lebanon", "Army", "Navy", 'Nato'
]










bummy = samp_sorted.loc[(samp_sorted['Year'] >= startyear)]
bummy.reset_index(inplace=True)
bummy.drop('index', axis=1, inplace=True)



df1 = pd.DataFrame(columns=['doc_name', 'date', 'transcript', 'president', 'title', 'Year'])


prev=startDate

for i in range(len(bummy)):
    date_string = bummy.loc[bummy.index[i], 'date']  # Accessing the 'date' column for each row
    cleaned_string = date_string.replace('-', '')
    
    # Convert string to number
    if 'T' in cleaned_string:
        cleaned_string=cleaned_string.split('T')[0]
    result = int(cleaned_string)
    if result<9999:
        
        original_str = str(prev)
        bluuuh = str(result)
    
        # Cut off the last four characters
        last_four = original_str[-4:]

        # Combine the remaining string with the new string
        res = bluuuh + last_four
        result =int(res)
        

    prev=result
    
    fumm=str(result)
    summmy=fumm[:4] + '-' + fumm[4:6] + '-' + fumm[6:]
    
    
    bummy.loc[bummy.index[i], 'date'] = summmy
    
    
    if startDate <= result <= endDate:
        
        # Assuming df1 is the DataFrame you want to add a row to, and df2 is the DataFrame from which you want to add the row
        row_to_add = bummy.loc[i]  # Select the row you want to add from df2

        df1.loc[len(df1)] = row_to_add.values
        
        
        
        
# Ensure the 'date' column is in datetime format
df1['date'] = pd.to_datetime(df1['date'])


# Create a DataFrame with each president's first and last speech date
president_dates = df1.groupby('president')['date'].agg(['min', 'max']).reset_index()
president_dates.columns = ['president', 'first_speech', 'last_speech']

# Print the new DataFrame
print(president_dates)





# Create an empty DataFrame to store the counts
fumpy = pd.DataFrame(columns=['president'] + gents)

# Iterate over each president in df1
for president in df1['president'].unique():
    # Initialize a dictionary to store the counts for each president
    president_counts = {'president': president}
    # Filter df1 for speeches by the current president
    president_df = df1[df1['president'] == president]
    # Iterate over each word in gents
    for word in gents:
        # Count the occurrences of the word in the current president's speeches
        word_count = sum(president_df['transcript'].str.count(word))
        # Add the count to the dictionary
        president_counts[word] = word_count
    # Append the counts for the current president to fumpy
    fumpy = fumpy.append(president_counts, ignore_index=True)
    
    
# Set the 'president' column as the index
fumpy.set_index('president', inplace=True)

# Calculate the sum of each column and sort the columns by the sum in descending order
fumpy = fumpy.loc[:, fumpy.sum().sort_values(ascending=False).index]


republican_presidents_df = pd.DataFrame({
    "Name": [
        "Donald Trump",
        "Dwight D. Eisenhower",
        "George H. W. Bush",
        "George W. Bush",
        "Gerald Ford",
        "Richard M. Nixon",
        "Ronald Reagan"
    ]
})

# Print the DataFrame fumpy
print(fumpy)


# In[249]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# Fill missing values with zeros
fumpy_filled = fumpy.fillna(0)

# Convert fumpy to numeric type
fumpy_numeric = fumpy_filled.apply(pd.to_numeric)





# Set the size of the heatmap
plt.figure(figsize=(18, 10))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_transposed, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 25  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx-0.0, y_pos_below -0.5, label, ha='center', va='center', fontsize=16, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+0.1, y_pos_below -0.05, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)

    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=135)
plt.ylabel('Entities', fontsize=15)
plt.title(f'Middle East Entity Frequency by Presidents \n {startyear} - {endyear} \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.15, "The heatmap displays the frequency of specific entities mentioned in President Speeches during the signifiacnt years of Middle East involvement, with annotations indicating the start and end dates of of presidents in office during that time period starts with the Navy establishing Middle East Force for first time and this period is still going on to this day so ends with the earliest speech in data. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()








# Initialize a StandardScaler
scaler = StandardScaler()

# Scale the data
fumpy_standardized = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(18, 10))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_standardized, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 25  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx-0.0, y_pos_below -0.5, label, ha='center', va='center', fontsize=16, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+0.1, y_pos_below -0.05, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)

legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=135)
plt.ylabel('Entities', fontsize=15)
plt.title(f'Middle East Entity Frequency by Presidents \n {startyear} - {endyear} Standardized\n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.15, "The heatmap displays the Standardized frequency of specific entities mentioned in President Speeches during the signifiacnt years of Middle East involvement, with annotations indicating the start and end dates of of presidents in office during that time period starts with the Navy establishing Middle East Force for first time and this period is still going on to this day so ends with the earliest speech in data. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()








# Initialize a MinMaxScaler
scaler = MinMaxScaler()

# Scale the data
fumpy_scaled = pd.DataFrame(scaler.fit_transform(fumpy_transposed), columns=fumpy_transposed.columns, index=fumpy_transposed.index)

# Set the size of the heatmap
plt.figure(figsize=(18, 10))  # You can adjust the size as needed

# Create the transposed DataFrame
fumpy_transposed = fumpy_numeric.T

# Create the heatmap without annotations first
sns.heatmap(fumpy_scaled, annot=False, cmap='Blues')

# Annotate each cell with the corresponding value
for i in range(len(fumpy_transposed.index)):
    for j in range(len(fumpy_transposed.columns)):
        plt.text(j + 0.5, i + 0.5, fumpy_transposed.iloc[i, j], ha='center', va='center', color='red', fontsize=17)

        
# Set custom x-axis tick colors
x_labels = fumpy_transposed.columns
x_colors = ['red' if label in republican_presidents_df['Name'].values else 'blue' for label in x_labels]


# Map the start and end dates for each president
date_mapping = {
    row['president']: (row['first_speech'].strftime('%Y-%m-%d'), row['last_speech'].strftime('%Y-%m-%d'))
    for _, row in president_dates.iterrows()
}


# Use plt.text to position the x-tick labels manually below the heatmap
y_pos_below = 25  # Adjust as needed to position labels below the heatmap
for idx, (label, color) in enumerate(zip(x_labels, x_colors)):
    plt.text(idx-0.0, y_pos_below -0.5, label, ha='center', va='center', fontsize=16, color=color, rotation=40)
    start_date, end_date = date_mapping[label]
    plt.text(idx+0.1, y_pos_below -0.05, f'{start_date} - {end_date}', ha='center', va='center', fontsize=12, color='black', rotation=40)

    
legend_elements = [
    Patch(facecolor='blue', edgecolor='blue', label='Democrat'),
    Patch(facecolor='red', edgecolor='red', label='Republican')
]


plt.legend(handles=legend_elements, 
           loc='lower left', 
           bbox_to_anchor=(-0.3, -0.2),  # Coordinates for 'bbox_to_anchor' are relative to the axes
           title="Fontcolor of Presidents",
           fontsize='small',  # Adjust text size
           title_fontsize='medium',  # Adjust title font size
           frameon=True,  # Toggle the frame
           shadow=True)  # Add shadow for better visibility   
    
    
    
    
    
plt.xticks(ticks=[], labels=[])  # Remove default xticks to avoid overlap
# Set y-axis tick labels (word list)
plt.yticks(ticks=range(len(fumpy_transposed.index)), labels=fumpy_transposed.index, fontsize=20)      
plt.xlabel('President', fontsize=20, labelpad=135)
plt.ylabel('Entities', fontsize=15)
plt.title(f'Middle East Entity Frequency by Presidents \n {startyear} - {endyear} MinMax Scaled \n', fontsize=25)

# Add a caption
plt.figtext(0.4, -0.15, "The heatmap displays the MinMax Scaled frequency of specific entities mentioned in President Speeches during the signifiacnt years of Middle East involvement, with annotations indicating the start and end dates of of presidents in office during that time period starts with the Navy establishing Middle East Force for first time and this period is still going on to this day so ends with the earliest speech in data. Red labels highlight Republican Presidents and Blue higlights Depmocraric Presidents.", wrap=True, horizontalalignment='center', fontsize=15)

plt.show()


# # Below is just additional Data visualizations that help with understanding data but are not a a final submission and dont want to be considered for grading

# In[265]:



import json
with open('Data_Processing/speeches.json', 'r') as file:
    json_data = json.load(file)

# Convert JSON data to a string
json_string = json.dumps(json_data, indent=4)  # indent is optional, for pretty-printing

# Load JSON data
with open('Data_Processing/speeches.json', 'r') as file:
    data = json.load(file)

# Normalize JSON data
normalized_data = pd.json_normalize(data)  # This assumes 'data' is a list of records

# Create DataFrame
df = pd.DataFrame(normalized_data)


# In[266]:


samp_sorted= pd.read_csv('Data_Processing/outputfinal_filename.csv')
samp_sorted


# In[267]:


get_ipython().system('pip install selenium')

import nltk

# Download the required NLTK corpora
nltk.download('state_union')

# Now, you can safely load and use the 'state_union' corpus
from nltk.corpus import state_union

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time, os
import random
import pickle

from nltk.probability import FreqDist
from nltk.corpus import state_union
import re
import string
from nltk.stem import LancasterStemmer
nltk.download('state_union')
zetemp=samp_sorted
zetemp.sort_values(by=['Year'], inplace=True)
zetemp.head(3)


# Text preprocessing steps - remove numbers, captial letters and punctuation

alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
no_n = lambda x: re.sub('\n', '', x)
no_r = lambda x: re.sub('\r', '', x)
no_hyphen = lambda x: re.sub('-', ' ', x)
zetemp['transcript'] = zetemp.transcript.map(alphanumeric).map(punc_lower).map(no_n).map(no_r).map(no_hyphen)
zetemp['Split'] = zetemp.transcript.str.split()
zetemp.head(3)


stemmer = LancasterStemmer()
zetemp['Stemmed'] = zetemp['Split'].apply(lambda x: [stemmer.stem(y) for y in x])# Stem every word.
zetemp.sample(3)



zetemp.to_csv('Data_Processing/Presidents_Speeches.csv')


zetemp.to_pickle("my_Presidents.pkl")

with open('my_Presidents.pkl', 'wb') as picklefile:
    pickle.dump(zetemp, picklefile)
    
    
with open("my_Presidents.pkl", 'rb') as picklefile: 
    my_old_df = pickle.load(picklefile)
    
zetemp = my_old_df
zetemp.reset_index(level = 0, inplace = True)
zetemp = df.rename(columns = {'index':'Title'})
zetemp.head(3) 


# In[268]:


zetemp.president.value_counts()


# In[269]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.figure(figsize=(8, 14))
my_colors = ['orange','orange','orange','orange','orange','orange', 'darkviolet', 'darkviolet', 'orange', 'orange','darkviolet', 'orange',
            'orange', 'orange', 'darkviolet','orange', 'orange','orange', 'orange', 'orange', 'darkviolet', 'darkviolet', 'orange','darkviolet',
             'orange','darkviolet', 'orange', 'orange','darkviolet', 'darkviolet','orange','darkviolet', 'orange', 'darkviolet',
             'orange', 'darkviolet','darkviolet','darkviolet','darkviolet','orange', 'orange', 'darkviolet','darkviolet','darkviolet','darkviolet']

df.president.value_counts(ascending=True).plot(kind = 'barh', color=my_colors)
plt.xlabel('Number of Speeches', fontsize = 15)
plt.ylabel('Presidents', fontsize = 16)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.title('Number of Speeches per President', fontsize = 20)
plt.tight_layout()
plt.savefig('No._Pres_Speeches.pdf');


# In[270]:


samp_sorted= pd.read_csv('Data_Processing/outputfinal_filename.csv')
samp_sorted

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('Data_Processing/outputfinal_filename.csv')
df


# Count the number of words in each transcript
df['word_count'] = df['transcript'].str.split().str.len()

# Group by decade (assuming 'Year' column is the last 4 digits of the 'date' column)
df['decade'] = df['date'].str[:4].astype(int) // 10 * 10

# Increase font size and set a single color
sns.set(style='whitegrid')
plt.figure(figsize=(20, 10))
sns.boxplot(x='decade', y='word_count', data=df, color='lightblue')  # Adjust the color as needed

# Customize plot appearance
plt.title('Word Count of Presidential Speeches Over Decades', fontsize=24)
plt.xlabel('Decade', fontsize=18)
plt.ylabel('Word Count', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(rotation=45)  # Rotate x labels if needed

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Messing Around with Data starting here to figure out how to represent the knowledge maps This part is for the relational entites analysis, Analysis is done but I failed in representing it properly, and nicely

# In[250]:


charted= pd.read_csv('Data_Processing/chart_ordered.csv')
charted


# In[251]:


months=['January', 'February', 'March', 'April', 'May', 'June', 
 'July', 'August', 'September', 'October', 'November', 'December']

count=0
for i in range(len(charted)):
    jack=str(charted.loc[i,'entity2'])
    for month in months:
        if month in jack :
            charted = charted.drop(i)
            #count+=1


# In[252]:


charted


# In[253]:


charted.reset_index(inplace=True)
charted.drop('index', axis=1, inplace=True)
charted


# In[254]:


charted.loc[charted['avg_sentiment']<-0.9]


# In[255]:


# reove counts and connections less than 30, reomove entite\y2 which are substrings of entity1
chartoy=charted.loc[charted['entity_count']>29]
chartoy=chartoy.loc[charted['connect_count']>29]
chartoy.reset_index(inplace=True)
chartoy.drop('index', axis=1, inplace=True)
chartoy = chartoy[~chartoy.apply(lambda row: row['entity2'] in row['entity1'], axis=1)]
chartoy.reset_index(inplace=True)
chartoy.drop('index', axis=1, inplace=True)
chartoy.to_csv('Data_Processing/chart_ordy.csv', index=False)
chartoy


# In[256]:


chartoy.loc[chartoy['avg_sentiment']<0]


# In[257]:


chartoy = chartoy.drop_duplicates(subset='entity1')
chartoy.reset_index(inplace=True)
chartoy.drop('index', axis=1, inplace=True)
chartoy


# In[258]:



original_rows_to_remove = [249, 245, 236, 232, 221, 188, 177, 160, 129, 128, 118, 115, 110, 105, 90, 84, 76, 75, 69, 56, 54, 52, 48, 39, 38, 36, 31, 24]

# Adjusting to 0-based indexing by subtracting 1 from each index
rows_to_remove = [x - 1 for x in original_rows_to_remove]

# Remove the specified rows
chartoy = chartoy.drop(rows_to_remove)

# Optional: Reset index if you want a continuous index after removal
chartoy = chartoy.reset_index(drop=True)


# In[259]:


chartoy


# ## Plotting Knowdlege map of the 

# In[260]:


def ploting_map(data_new, stwing= "Knowledge Map of Entity Relationships"):
    # Creating a new graph
    G_new = nx.Graph()

    # Extracting all unique entities
    all_entities_new = set(data_new['entity1']).union(set(data_new['entity2']))

    # Creating a dictionary for entity sizes
    default_size = 100  # Default size
    entity_sizes_new = {entity: data_new[data_new['entity1'] == entity]['entity_count'].iloc[0]
                        if entity in data_new['entity1'].values else default_size 
                        for entity in all_entities_new}

    # Adding nodes with their sizes, scaled 5 times bigger
    for entity, size in entity_sizes_new.items():
        G_new.add_node(entity, size=size * 50)  # Scaling the node size by 5 times

    # Adding edges with varying thickness and color based on sentiment
    for _, row in data_new.iterrows():
        width = abs(row['avg_sentiment']) * 10  # Scale the width
        color = 'red' if row['avg_sentiment'] < 0 else 'blue'
        G_new.add_edge(row['entity1'], row['entity2'], weight=width, color=color)

    # Extracting sizes for nodes, scaled 5 times bigger
    sizes_new_scaled = [G_new.nodes[node]['size']/200 for node in G_new.nodes]

    # Extracting colors and weights for edges
    edge_colors_new = nx.get_edge_attributes(G_new,'color').values()
    edge_weights_new = nx.get_edge_attributes(G_new,'weight').values()

    pos = nx.spring_layout(G_new, k=1.5, iterations=20) 

    # Drawing the graph with scaled node sizes
    plt.figure(figsize=(10, 8))
    nx.draw(G_new,pos, with_labels=True, node_size=sizes_new_scaled, width=list(edge_weights_new), edge_color=list(edge_colors_new), node_color='skyblue', edgecolors='black')
    plt.title(stwing)
    plt.show()


# # The following Data Visulizations are just in testing and progress, not fully developed as it is too messy to be a finalized datavisual

# In[261]:


ploting_map(charted.loc[charted['avg_sentiment']<0][:100], "100-150 negative sentiment over .75")


# In[262]:


ploting_map(charted.loc[charted['avg_sentiment']<0][:400], "First 400 negative Sent. Relation Knowledge Map")


# In[263]:


ploting_map(chartoy.loc[chartoy['avg_sentiment']<0], "all negative Sent. with counts more than 30")


# In[264]:


ploting_map(chartoy.loc[chartoy['avg_sentiment']>0.25], "top positive Sent. of 0ver 0.25 Relation Knowledge Map top relation for each entity")


# In[ ]:





# In[ ]:





# In[ ]:




