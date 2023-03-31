#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install missingno


# In[2]:


pip install plotly


# In[3]:


pip install catboost


# In[4]:


pip install xgboost


# In[5]:


pip install pyo


# In[6]:


import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/datazng/telecom-company-churn-rate-call-center-data")


# In[7]:


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyo
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[8]:


#filename
file_CustomerInfo=('Telecom Churn Rate Dataset.xlsx')
file_Call_CentreInfo=('Telecom Company Call-Center-Dataset.xlsx')


# In[9]:


def read_data(filename):
    df=pd.read_excel(filename)
    return df


# In[10]:


# Read the xlsx file into a dataframe
df_CustomerInfo=read_data(file_CustomerInfo)
df_CustomerInfo.head()


# In[11]:


#Descriptive status of the df
def decrip_data(df):
    print(df.info())
    print(df.shape)
    print(df.describe())
    return


# In[12]:


decrip_data(df_CustomerInfo)


# In[13]:


#Data Cleaning

def clean_data(df):

    # drop missing values
    print(df.isna().sum())
    df = df.dropna()
    # Check for duplicates in the entire DataFrame
    duplicates = df.duplicated()
    df.drop_duplicates(inplace=True)
    # Visualize missing values as a matrix
    msno.matrix(df);
    return df
    


# In[14]:


clean_data(df_CustomerInfo)


# In[15]:


# Data Manipulation

#df_CustomerInfo = df_CustomerInfo.drop(['customerID'])
# Map values to "No" or "Yes"
df_CustomerInfo["SeniorCitizen"]= df_CustomerInfo["SeniorCitizen"].map({0: "No", 1: "Yes"})
df_CustomerInfo['OnlineSecurity'] = df_CustomerInfo['OnlineSecurity'].replace('No internet service', 'No')
df_CustomerInfo['OnlineBackup'] = df_CustomerInfo['OnlineBackup'].replace('No internet service', 'No')
df_CustomerInfo['DeviceProtection'] = df_CustomerInfo['DeviceProtection'].replace('No internet service', 'No')
df_CustomerInfo['TechSupport'] = df_CustomerInfo['TechSupport'].replace('No internet service', 'No')
df_CustomerInfo['StreamingTV'] = df_CustomerInfo['StreamingTV'].replace('No internet service', 'No')
df_CustomerInfo['StreamingMovies'] = df_CustomerInfo['StreamingMovies'].replace('No internet service', 'No')


df_CustomerInfo['TotalCharges'] = pd.to_numeric(df_CustomerInfo.TotalCharges, errors='coerce')
df_CustomerInfo.isnull().sum()
df_CustomerInfo[np.isnan(df_CustomerInfo['TotalCharges'])]
df_CustomerInfo[df_CustomerInfo['tenure'] == 0].index
df_CustomerInfo.drop(labels=df_CustomerInfo[df_CustomerInfo['tenure'] == 0].index, axis=0, inplace=True)
df_CustomerInfo[df_CustomerInfo['tenure'] == 0].index
df_CustomerInfo.fillna(df_CustomerInfo["TotalCharges"].mean())
df_CustomerInfo.isnull().sum()
df_CustomerInfo.head()
df_CustomerInfo["InternetService"].describe(include=['object', 'bool'])
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_CustomerInfo[numerical_cols].describe()


# In[16]:


#Dataset is clean now

#EDA CustomerInfo


# In[17]:


#What is the overall churn rate of the telecom company and how does it vary by customer demographics(e.g gender)
g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=df_CustomerInfo['gender'].value_counts(), name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=df_CustomerInfo['Churn'].value_counts(), name="Churn"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.2, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[18]:


df_CustomerInfo["Churn"][df_CustomerInfo["Churn"]=="No"].groupby(by=df_CustomerInfo["gender"]).count()


# In[19]:


df_CustomerInfo["Churn"][df_CustomerInfo["Churn"]=="Yes"].groupby(by=df_CustomerInfo["gender"]).count()


# In[20]:


plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors =['#e75480','#9370db']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[21]:


#The overall churn rate of the telecom company is 26.54%. The churn rate varies by customer demographics such as gender, partner status, etc.

#Based on the analysis, the churn rate is higher among female customers (26.92%) compared to male customers (26.16%). The churn rate is also significantly higher among customers who do not have a partner (32.96%) compared to those who have a partner (19.66%).

#There is negligible difference in customer percentage/ count who chnaged the service provider. Both genders behaved in similar fashion when it comes to migrating to another service provider/firm.


# In[22]:


df=df_CustomerInfo.copy()


# In[23]:


#q2.Which services are most commonly subscribed to by customers?
services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


df['InternetService'] = df['InternetService'].replace('DSL', 'Yes')
df['InternetService'] = df['InternetService'].replace('Fiber optic', 'Yes')
# create a DataFrame with the service counts
service_counts = {}
for service in services:
    service_counts[service] = df[service].value_counts().to_dict()
df_service_counts = pd.DataFrame(service_counts)

# convert the DataFrame from wide to long format
df_service_counts = df_service_counts.reset_index().rename(columns={'index': 'Subscribed'})
df_service_counts = pd.melt(df_service_counts, id_vars=['Subscribed'], var_name='Services', value_name='Count')

# create a grouped bar chart
fig = px.bar(df_service_counts, x='Services', y='Count', color='Subscribed', 
            # color_discrete_map={'Yes': '#66b3ff', 'No': '#ff6666'},
             title='Service Subscription Count', barmode='group')

fig.update_layout(xaxis_title='Services', yaxis_title='Count')
fig.show()


# In[24]:


#Which services are most commonly subscribed to by customers - Phone Service,followed by Internet Service


# In[25]:


fig = px.histogram(df_CustomerInfo, x="Contract", color="Churn", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[26]:


#About 75% of customer with Month-to-Month Contract opted to move out as compared to 13% of customrs with One Year Contract and 3% with Two Year Contract


# In[27]:


labels = df_CustomerInfo['PaymentMethod'].unique()
values = df_CustomerInfo['PaymentMethod'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>Payment Method Distribution</b>")
fig.show()


# In[28]:


fig = px.histogram(df_CustomerInfo, x="PaymentMethod", color="Churn", barmode="group",title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[29]:


#Major customers who moved out were having Electronic Check as Payment Method.
#Customers who opted for Credit-Card automatic transfer or Bank Automatic Transfer and Mailed Check as Payment Method were less likely to move out.


# In[30]:


df_CustomerInfo["InternetService"].unique()


# In[31]:


df_CustomerInfo[df_CustomerInfo["gender"]=="Male"][["InternetService", "Churn"]].value_counts()


# In[32]:


df_CustomerInfo[df_CustomerInfo["gender"]=="Female"][["InternetService", "Churn"]].value_counts()


# In[33]:


fig = go.Figure()

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [969, 993, 219, 240],
  name = 'DSL',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [889, 910, 664, 633],
  name = 'Fiber optic',
))

fig.add_trace(go.Bar(
  x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
       ["Female", "Male", "Female", "Male"]],
  y = [691, 722, 56, 57],
  name = 'No Internet',
))

fig.update_layout(title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>")

fig.show()


# In[34]:


#A lot of customers choose the Fiber optic service and it's also evident that the customers who use Fiber optic have high churn rate, this might suggest a dissatisfaction with this type of internet service.
#Customers having DSL service are majority in number and have less churn rate compared to Fibre optic service.


# In[35]:


#color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df_CustomerInfo, color="Churn", x="Dependents", barmode="group", title="<b>Dependents distribution</b>"
                  )
#, color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[36]:


#Customers without dependents are more likely to churn


# In[37]:


#color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df_CustomerInfo, color="Churn", x="Partner", barmode="group", title="<b>Chrun distribution w.r.t. Partners</b>"
                  )#, color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[38]:


#Customers that doesn't have partners are more likely to churn


# In[39]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df_CustomerInfo, color="Churn", x="SeniorCitizen", title="<b>Chrun distribution w.r.t. Senior Citizen</b>"
                   #, color_discrete_map=color_map
                  )
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[40]:


#It can be observed that the fraction of senior citizen is very less.
#Most of the senior citizens churn.


# In[41]:


color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
fig = px.histogram(df_CustomerInfo, color="Churn", x="OnlineSecurity", barmode="group", title="<b>Churn w.r.t Online Security</b>"
                   #, color_discrete_map=color_map
                  )
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[42]:


#Most customers churn in the absence of online security,


# In[43]:


color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
fig = px.histogram(df_CustomerInfo, color="Churn", x="PaperlessBilling",  title="<b>Chrun distribution w.r.t. Paperless Billing</b>"
                   #, color_discrete_map=color_map
                  )
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[44]:


#Customers with Paperless Billing are most likely to churn.


# In[45]:


fig = px.histogram(df_CustomerInfo, color="Churn", x="TechSupport",barmode="group",  title="<b>Chrun distribution w.r.t. TechSupport</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[46]:


#Customers with no TechSupport are most likely to migrate to another service provider.


# In[47]:


color_map = {"Yes": '#00CC96', "No": '#B6E880'}
fig = px.histogram(df_CustomerInfo, color="Churn", x="PhoneService", title="<b>Chrun distribution w.r.t. Phone Service</b>"
                   #, color_discrete_map=color_map
                  )
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[48]:


#Very small fraction of customers don't have a phone service and out of that, 1/3rd Customers are more likely to churn.


# In[49]:


sns.set_context("paper",font_scale=1.1)
ax = sns.kdeplot(df_CustomerInfo.MonthlyCharges[(df_CustomerInfo["Churn"] == 'No') ],
                color="Blue", shade = True);
ax = sns.kdeplot(df_CustomerInfo.MonthlyCharges[(df_CustomerInfo["Churn"] == 'Yes') ],
                ax =ax, color="Red", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of monthly charges by churn');


# In[50]:


#Customers with higher Monthly Charges are also more likely to churn


# In[51]:


df_CustomerInfo['TotalCharges'] = pd.to_numeric(df_CustomerInfo['TotalCharges'], errors='coerce')

ax = sns.kdeplot(df_CustomerInfo.TotalCharges[(df_CustomerInfo["Churn"] == 'No') ],
                color="Blue", shade = True);
ax = sns.kdeplot(df_CustomerInfo.TotalCharges[(df_CustomerInfo["Churn"] == 'Yes') ],
                ax =ax, color="Red", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of total charges by churn');


# In[52]:


fig = px.box(df_CustomerInfo, x='Churn', y = 'tenure',color='Churn')

# Update yaxis properties
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Tenure vs Churn</b>',
)

fig.show()


# In[53]:


#New customers are more likely to churn


# In[54]:


plt.figure(figsize=(25, 10))

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='YlGnBu', vmin=-1, vmax=1)


# In[55]:


#The three features that are most positively correlated with Churn are:

#numTechTickets
#Senior Citizen
#Partner
#Online Back

#The features that are most negatively correlated with Churn are:

#Contact
#Payment Method
#PaperlessBilling
#OnlineSecurity
#TechSupport


# In[56]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)


# In[57]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df_CustomerInfo)


# In[58]:


#Dataset-2


# In[59]:


df_Call_CentreInfo=read_data(file_Call_CentreInfo)
df_Call_CentreInfo.head()


# In[60]:


decrip_data(df_Call_CentreInfo)
clean_data(df_Call_CentreInfo)


# In[61]:


#What is the overall satisfaction rate of customers?
satisfaction_rate = df_Call_CentreInfo['Satisfaction rating'].mean()
print("Overall satisfaction rate: {:.2f}".format(satisfaction_rate))


# In[62]:


# Group the data by 'Topic' and 'Agent' and calculate the mean satisfaction rating for each group
satisfaction_by_topic = df_Call_CentreInfo.groupby(['Topic'])['Satisfaction rating'].mean()

# Print the results
print(satisfaction_by_topic)


# In[63]:


#Based on the data provided, it appears that customers were generally satisfied with all topics, with an average satisfaction rating of around 3.4 to 3.4. Admin Support had the highest satisfaction rating at 3.43, followed by Streaming at 3.40, Technical Support at 3.41, Payment related at 3.40, and Contract related at 3.38.


# In[64]:


#q3.Which topics have the highest and lowest resolution rates? 
# Use boolean indexing to extract the rows with resolved == 'Y'
resolved_y = df_Call_CentreInfo[df_Call_CentreInfo['Resolved'] == 'Y']

# Calculate the number of rows for each topic
resolved_y_counts = resolved_y["Topic"].value_counts()

# Print the extracted rows
print(resolved_y_counts)

resolved_n = df_Call_CentreInfo[df_Call_CentreInfo['Resolved'] == 'N']

# Calculate the number of rows for each topic
resolved_n_counts = resolved_n["Topic"].value_counts()

# Print the extracted rows
print(resolved_n_counts)



# In[65]:


#Based on the given data, the topic with the highest resolution rate is "Streaming" with 749 resolved out of 800 total, and the topic with the lowest resolution rate is "Contract related" with 709 resolved out of 800 total.
#The result shows the number of unresolved cases for each topic, ordered from highest to lowest. It appears that Technical Support has the highest number of unresolved cases, followed by Payment related, Streaming, Contract related, and Admin Support.


# In[66]:



#What is the average speed of answer for different topics and agents?
import plotly.express as px

# Calculate average speed of answer by topic and agent
avg_soa = df_Call_CentreInfo.groupby(['Topic', 'Agent'])['Speed of answer in seconds'].mean().reset_index()

# get the index of the row with the maximum speed for each topic
idx = avg_soa.groupby('Topic')['Speed of answer in seconds'].idxmax()

# get the agent name for each row using the index
result = avg_soa.loc[idx, ['Topic', 'Agent']]

# print the result
print(result)


# Create plotly visualization
fig = px.bar(avg_soa, x='Topic', y='Speed of answer in seconds', color='Agent', title='Average Speed of Answer by Topic and Agent')
fig.show()


# In[67]:


#                Topic   Agent
#6       Admin Support  Martha
#13   Contract related     Joe
#19    Payment related    Greg
#30          Streaming  Martha
#37  Technical Support     Joe


# In[68]:


#q4.Which agents have the highest and lowest resolution rates?
# Filter the rows with resolved == Y
resolved_df = df_Call_CentreInfo[df_Call_CentreInfo["Resolved"] == "Y"]

# Calculate the number of resolved calls for each agent
resolved_counts = resolved_df["Agent"].value_counts()

# Calculate the total number of calls for each agent
total_counts = df_Call_CentreInfo["Agent"].value_counts()

# Calculate the resolution rate for each agent
resolution_rates = resolved_counts / total_counts

# Sort the agents by resolution rate
sorted_agents = resolution_rates.sort_values()

# Print the agents with the highest and lowest resolution rates
print("Agents with highest resolution rates:")
print(sorted_agents.tail())
print("\nAgents with lowest resolution rates:")
print(sorted_agents.head())


# In[69]:


#The agents with the highest resolution rates are listed in decreasing order: Stewart, Greg, Becky, Joe, and Dan. The agents with the lowest resolution rates are listed in increasing order: Diane, Martha, Jim, Stewart (note that this agent appears on both lists), and Greg.


# In[70]:


#What is the overall satisfaction rating for calls related to Admin Support?
#What is the overall satisfaction rating for calls related to Tech Support?

# Filter the DataFrame to include only calls related to Admin Support
support_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Admin Support']

# Calculate the overall satisfaction rating for Admin Support calls
overall_satisfaction_rating = support_calls['Satisfaction rating'].mean()

print("The overall satisfaction rating for calls related to Admin Support is:", overall_satisfaction_rating)

# Filter the DataFrame to include only calls related to Tech Support
support_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Technical Support']

# Calculate the overall satisfaction rating for Admin Support calls
overall_satisfaction_rating = support_calls['Satisfaction rating'].mean()

print("The overall satisfaction rating for calls related to Tech Support is:", overall_satisfaction_rating)


# In[71]:


#The overall satisfaction rating for calls related to Admin Support is: 3.4264150943396228
#The overall satisfaction rating for calls related to Tech Support is: 3.414906832298137


# In[72]:


#What is the distribution of satisfaction ratings for calls related to Admin Support? represent using daigram

# Filter the DataFrame to include only calls related to Admin Support
admin_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Admin Support']

# Create a histogram of the satisfaction ratings for Admin Support calls
#plt.hist(admin_calls['Satisfaction rating'], bins=5, range=(1, 5),color='r')
ax = sns.distplot(admin_calls['Satisfaction rating'], color= 'r')
# Add labels and a title to the plot
plt.xlabel('Satisfaction Rating')
plt.ylabel('Count')
plt.title('Distribution of Satisfaction Ratings for Admin Support Calls')

# Show the plot
plt.show()


# In[73]:


# Filter the DataFrame to include only calls related to Admin Support
admin_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Technical Support']

# Create a histogram of the satisfaction ratings for Admin Support calls
#plt.hist(admin_calls['Satisfaction rating'], bins=5, range=(1, 5),color='r')
ax = sns.distplot(admin_calls['Satisfaction rating'], color= 'r')
# Add labels and a title to the plot
plt.xlabel('Satisfaction Rating')
plt.ylabel('Count')
plt.title('Distribution of Satisfaction Ratings for Technical Support Calls')

# Show the plot
plt.show()


# In[74]:


print(df_Call_CentreInfo)


# In[75]:


# convert the Duration column to datetime format
df_Call_CentreInfo['AvgTalkDuration'] = pd.to_datetime(df_Call_CentreInfo['AvgTalkDuration'], format='%H:%M:%S')

# extract the minute and second components of the Duration column
df_Call_CentreInfo['AvgTalkDuration'] = df_Call_CentreInfo['AvgTalkDuration'].dt.minute*60 + df_Call_CentreInfo['AvgTalkDuration'].dt.second

# format the Duration column as two decimal places
#df_Call_CentreInfo['AvgTalkDuration'] = df_Call_CentreInfo['AvgTalkDuration'].map('{:.2f}'.format)


# In[76]:


# filter data for calls related to Admin Support
admin_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Admin Support']
admin_calls = admin_calls[df_Call_CentreInfo['Resolved'] == 'Y']
admin_calls=admin_calls.dropna()


# In[77]:


admin_calls['AvgTalkDuration'] = pd.to_numeric(admin_calls['AvgTalkDuration'])


# In[78]:


talk_duration_avg=admin_calls['AvgTalkDuration'].mean()
asa_avg = admin_calls['Speed of answer in seconds'].mean()
# print results
print("Average Speed of Answer:", asa_avg)
print("Average Talk Duration:", talk_duration_avg)


# In[79]:


# Create a scatter plot
sns.scatterplot(x='Speed of answer in seconds', y='AvgTalkDuration', hue='Satisfaction rating', data=admin_calls)

# Calculate the correlation coefficient
corr = admin_calls['Speed of answer in seconds'].corr(admin_calls['AvgTalkDuration'])
print(f"Correlation coefficient: {corr}")


# In[80]:


#least correaltion between Avg talk duration and Speed of answers in sec


# In[81]:


# filter data for calls related to Admin Support
admin_calls = df_Call_CentreInfo[df_Call_CentreInfo['Topic'] == 'Technical Support']
admin_calls = admin_calls[df_Call_CentreInfo['Resolved'] == 'Y']
admin_calls=admin_calls.dropna()


# In[82]:


admin_calls['AvgTalkDuration'] = pd.to_numeric(admin_calls['AvgTalkDuration'])


# In[83]:


talk_duration_avg_tech=admin_calls['AvgTalkDuration'].mean()
asa_avg_tech = admin_calls['Speed of answer in seconds'].mean()

# print results
print("Average Speed of Answer:", asa_avg_tech)
print("Average Talk Duration:", talk_duration_avg_tech)


# In[84]:


import seaborn as sns

# create bar chart
sns.barplot(x=['Admin Support', 'Technical Support'],
            y=[asa_avg, asa_avg_tech])


# add title
plt.title('Average Speed of Answer')

# show plot
plt.show()


# In[85]:


import seaborn as sns

# create bar chart
sns.barplot(x=['Admin Support', 'Technical Support'],
            y=[talk_duration_avg, talk_duration_avg_tech])


# add title
plt.title('Average Talk Duration')

# show plot
plt.show()


# In[86]:


# Create a scatter plot
sns.scatterplot(x='Speed of answer in seconds', y='AvgTalkDuration', hue='Satisfaction rating', data=admin_calls)

# Calculate the correlation coefficient
corr = admin_calls['Speed of answer in seconds'].corr(admin_calls['AvgTalkDuration'])
print(f"Correlation coefficient: {corr}")


# In[87]:


# Building the test and train datasets


# In[88]:


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# In[89]:


df = df_CustomerInfo


# In[90]:


df = df.apply(lambda x: object_to_int(x))


# In[91]:


plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)


# In[92]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)


# In[94]:


def distplot(feature, frame, color='b'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)


# In[95]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)


# In[96]:


#Standardizing numeric attributes
df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std, color='b')


# In[97]:


# Divide the columns into 3 categories, one ofor standardisation, one for label encoding and one for one hot encoding
cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # those that need one-hot encoding
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #those that need label encoding


# In[98]:


scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[99]:


#Machine Learning Model and Evaluation

#1.KNN
#2.Random forest
#3.Logistic Regression
#4.Decison Tree Classifier
#5.Support Vector Classifier
#6.Naive Bayes Classifier
#7.XGBoost classifier
#8.Gradientboost Classifier


# In[100]:


def print_confusion_matrix(title,test_var,train_var):
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(test_var, train_var),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
    plt.title(title,fontsize=14)
    plt.show()
    return


# In[101]:


def print_roc_curve(title,label,test_var_X,test_var_y,model):
    y_pred_prob = model.predict_proba(test_var_X)[:,1]
    fpr_rf, tpr_rf, thresholds = roc_curve(test_var_y, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--' )
    plt.plot(fpr_rf, tpr_rf, label=label,color = "r")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title,fontsize=16)
    plt.show();
    return


# In[102]:


#KNN
knn_model = KNeighborsClassifier(n_neighbors = 11) 
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN accuracy:",accuracy_knn)
print(classification_report(y_test, predicted_y))

#Confusion matrix
print("KNN Confuson Matrix")
print(confusion_matrix(y_test, predicted_y))
print_confusion_matrix("KNN Confuson Matrix",y_test,predicted_y)

#ROC Curve
print_roc_curve("KNN ROC Curve","KNN",X_test,y_test,knn_model)


# In[103]:


#Random forest
model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print ("Random Forest accuracy:",metrics.accuracy_score(y_test, prediction_test))
print(classification_report(y_test, prediction_test))
#Confusion matrix
print("Random Forest Confuson Matrix")
print(confusion_matrix(y_test, prediction_test))
print_confusion_matrix("Random Forest Confuson Matrix",y_test,prediction_test)


#ROC Curve
auc_roc = roc_auc_score(y_test, prediction_test)
print('AUC-ROC score:', auc_roc)
print_roc_curve("Random Forest ROC Curve","Random Forest",X_test,y_test,model_rf)


# In[104]:


#Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)
lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)
#Confusion matrix
print("Logistic Regression Confuson Matrix")
print(confusion_matrix(y_test, lr_pred))
print_confusion_matrix("Logistic Regression Confuson Matrix",y_test,lr_pred)

#ROC Curve
auc_roc = roc_auc_score(y_test, lr_pred)
print('AUC-ROC score:', auc_roc)
print_roc_curve("Logistic Regression ROC Curve","Logistic Regression",X_test,y_test,lr_model)


# In[105]:


#Decison Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)

# Evaluate the model
print("Decision Tree accuracy is :",accuracy_dt)
print(classification_report(y_test, predictdt_y))

#Confusion matrix
print("Decision Tree Confuson Matrix")
print(confusion_matrix(y_test, predictdt_y))
print_confusion_matrix("Decision Tree Confuson Matrix",y_test,predictdt_y)

#ROC Curve
print_roc_curve("Decision Tree ROC Curve","Decision Tree",X_test,y_test,dt_model)


# In[106]:


#SVC
svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)
print(classification_report(y_test, predict_y))

#Confusion matrix
print("SVC Confuson Matrix")
print(confusion_matrix(y_test, predict_y))
print_confusion_matrix("SVC Confuson Matrix",y_test,predict_y)

#ROC Curve
#print_roc_curve("SVC ROC Curve","Decision Tree",X_test,y_test,svc_model)


# In[107]:


#Naive Bayes Classifier

# Create an Naive Bayes Classifier object
model = GaussianNB()

model.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Classifier Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

#Confusion matrix
print("Naive Bayes Classifier Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print_confusion_matrix("Naive Bayes Classifier Confusion Matrix",y_test,y_pred)

#ROC Curve
print_roc_curve("Naive Bayes Classifier ROC Curve","Naive Bayes Classifier",X_test,y_test,model)


# In[108]:


#XG Boost

# Create an XGBoost classifier object
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('XG Boost Accuracy:', accuracy)
print(classification_report(y_test, y_pred))

#Confusion matrix
print("XG Boost Confuson Matrix")
print(confusion_matrix(y_test, y_pred))
print_confusion_matrix("XG Boost Confuson Matrix",y_test,y_pred)

#ROC Curve
print_roc_curve("XG Boost ROC Curve","XG Boost",X_test,y_test,xgb_model)


# In[109]:


#Gradientboost Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting Classifier", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))
#Confusion matrix
print("Gradient Boosting Confuson Matrix")
print(confusion_matrix(y_test, gb_pred))
print_confusion_matrix("Gradient Boosting Confuson Matrix",y_test,gb_pred)
#ROC Curve
auc_roc = roc_auc_score(y_test, gb_pred)
print('AUC-ROC score:', auc_roc)
print_roc_curve("Gradient Boosting ROC Curve","Gradient Boosting",X_test,y_test,gb)


# In[110]:


#Based on the evaluation results, the best performing models in terms of accuracy are the Random Forest Classifier and the Logistic Regression Classifier with accuracy scores of 0.86.
#The worst performing model is the Support Vector Classifier (SVC) with an accuracy score of 0.73.
#Based on the precision, recall and F1-score, the best model is Logistic Regression, with a precision of 0.75 and recall of 0.72 for identifying the churned customers. The Random Forest Classifier also performed well, with a precision of 0.79 and recall of 0.64. However, it's worth noting that the performance of the models might vary based on the specific needs of the problem and the costs associated with false positives and false negatives.
#The AUC-ROC score for the Logistic Regression model is 0.8174 and the AUC-ROC score for the Random Forest Classifier is 0.7905. Therefore, the Logistic Regression model has a higher AUC-ROC score compared to the Random Forest Classifier, indicating that the Logistic Regression model has a better overall performance in terms of correctly identifying true positive and true negative cases.
#Overall Logistic Regression Classifier seem to be the best models for this task based on the evaluation results.

