#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset (Replace this with something more specific!)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Tip**: In this section of the report, provide a brief introduction to the dataset you've selected for analysis. At the end of this section, describe the questions that you plan on exploring over the course of the report. Try to build your report around the analysis of at least one dependent variable and three independent variables.
# >
# > If you haven't yet selected and downloaded your data, make sure you do that first before coming back here. If you're not sure what questions to ask right now, then make sure you familiarize yourself with the variables and the dataset context for ideas of what to explore.

# In[2]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import numpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[3]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


## This function allows me to get an estimate of how many values are in each category and 
## thus, whether I need to clean up any data.
df.nunique()


# In[8]:


# check for any missing or null values in dataset 
df.isnull().sum()


# In[9]:


# check to make sure there is no duplicated data
print(sum(df.duplicated()))


# In[10]:


# This will allow me to check for the parameters of the Age column so that 
# we can use it for a later exploration
df.Age.max()


# In[11]:


df.Age.min()


# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.
# 
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
# 
# ### Data Cleaning (Replace this with more specific notes!)

# In[12]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.

# Query where the Age is -1.
df.query('Age == "-1"')


# In[13]:


df = df.drop(df.index[99832])


# In[14]:


df.info()


# In[15]:


"""
Clean up the column names
"""
# Rename columns so that each column has same format
df.rename(columns={'PatientId':'Patient_id', 'AppointmentID':'Appointment_id','ScheduledDay':'Scheduled_day',
                   'AppointmentDay':'Appointment_day', 'Neighbourhood':'Neighborhood', 'Hipertension':'Hypertension',
                   'Handcap':'Handicap','No-show':'No_show'},
inplace=True)
# confirm changes
df.head()


# In[16]:


df['No_show'].replace({'Yes':1,'No':0}, inplace = True)
df.head()


# In[17]:


df['Scheduled_day']= pd.to_datetime(df['Scheduled_day'])
df['Appointment_day']= pd.to_datetime(df['Appointment_day'])
df.info()


# In[18]:


df.info()


# In[19]:


df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables.
# 
# ### Research Question 1 what is the percentage for showed up patients against not showed up patients?

# In[20]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
"""
This function will determine how many showed up for appointments and how many didn't. This function will be
compared to all three of the questions we will be asking.
"""
Not_showed = (df['No_show'] == 1).value_counts()
print( (df['No_show'] == 1).value_counts())

print((Not_showed[1] / Not_showed.sum()) * 100, '%')
labels = 'showed','not showed'
explode=(0,0.1)
pieGraph = Not_showed.plot.pie(figsize=(8,8),labels= labels,explode=explode, autopct='%1.1f%%', fontsize = 1);
pieGraph.set_title('showed and not showed percentage\n', fontsize = 15);
plt.legend();


# ### Research Question 2  what is the correlation between the features( SMS_received, Scholarship , Age ,Gender ,Alcoholism) and (show up, no show up patients )

# Using a bar chart to demonstrate relation between SMS received and showed up patients.

# In[21]:


#create a variable for patients who showed up and not show up.

showed_up = df['No_show'] == 0
not_show_up = df['No_show'] == 1
df['showed_up'] = showed_up
df['not_show_up'] = not_show_up


# In[22]:


# using groupby function to show relation between SMS received feature and showed up percentage. 
# visualize the average of patients who received SMS and not.
print(df.groupby ('SMS_received')['showed_up'].mean())
df.groupby ('SMS_received')['showed_up'].mean().plot(kind='bar', figsize=(8,8))
plt.title("relation between patients who have recevied an sms and patients who showed")
plt.xlabel('showed')
plt.ylabel('SMS received')
plt.legend()


# Using a bar chart to demonstrate relation between scholarship and showed up patients

# In[23]:


print(df.groupby ('Scholarship')['showed_up'].mean())
df.groupby ('Scholarship')['showed_up'].mean().plot(kind='bar', figsize=(8,8))
plt.title("relation between patients who have scolarship and patients who showed")
plt.xlabel('showed')
plt.ylabel('Scholarship')
plt.legend()


# Using a histogram to show the correlation between Age and showed up patients.

# In[24]:


# using groupby function to show relation between Age feature and showed up percentage. 
# visualize the average of patients who showed up.

df.groupby ('Age')['showed_up'].mean().hist(bins=5,label='showed');
plt.title("relation between patients age and patients who showed")
plt.xlabel('showed')
plt.ylabel('Age')
plt.legend()


# In[25]:


#by using mean function.
df.Age[showed_up].mean()


# In[26]:


# using groupby function to show relation between Age feature and not show up percentage. 
# visualize the average of patients who did not show up.
df.groupby ('Age')['not_show_up'].mean().hist(bins=5, label='not showed');
plt.title("relation between  age and patients who not showed")
plt.xlabel('not showed')
plt.ylabel('Age')
plt.legend()


# In[27]:


df.Age[not_show_up].mean()


# Using a histogram to show the correlation between Age and not show up patients.

# In[28]:


# using groupby function to show relation between Gender feature and showed up percentage. 
# visualize the average of patients who is male or female.
print(df.groupby ('Gender')['No_show'].mean())
df.groupby ('Gender')['showed_up'].mean().plot(kind='bar', figsize=(8,8))
plt.title("relation between  gender and patients who showed")
plt.xlabel('showed')
plt.ylabel('Gender')
plt.legend()


# In[29]:


# using groupby function to show relation between Gender feature and showed up percentage. 
# visualize the average of patients who is male or female.
print(df.groupby ('Alcoholism')['No_show'].mean())
df.groupby ('Alcoholism')['showed_up'].mean().plot(kind='bar', figsize=(8,8))
plt.title("relation between Alcoholism and patients who showed")
plt.xlabel('showed')
plt.ylabel('Alcoholism')
plt.legend()


# ### Question 3: What are the relationships between the three variables (Alcoholism,Scholarship and SMS_Received)?

# Scholarship and SMS_Received

# In[31]:


# Plot a bar graph based on groupby data
groups = df.groupby(['Scholarship','SMS_received']).size().unstack(fill_value=0)
groups.plot.bar()

# Set titles and axes
plt.title('relation between Scholarship and SMS_received', fontsize = 20)
plt.xlabel('Scholarship', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)

# use the magic word to show the bar graph
plt.show()


# There is a direct relationship between scholarships and SMS_received. As scholarships increased, text messages received increased. 

# Alcoholism and SMS_received

# In[32]:


groups = df.groupby(['Alcoholism','SMS_received']).size().unstack(fill_value=0)
groups.plot.bar()

# Set titles and axes
plt.title('relation between Alcoholism and SMS_received', fontsize = 20)
plt.xlabel('Alcoholism', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)

# use the magic word to show the bar graph
plt.show()


# There is a direct relationship between alcoholism and SMS_received. As alcoholism increased, text messages received increased.

# Alcoholism and Scholarships

# In[33]:


# Plot a bar graph based on groupby data
groups = df.groupby(['Scholarship','Alcoholism']).size().unstack(fill_value=0)
groups.plot.bar()

# Set titles and axes
plt.title('relation between Alcoholism and Scholarships', fontsize = 20)
plt.xlabel('Scholarship', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)

# use the magic word to show the bar graph
plt.show()


# There is an indirect relationship between alcoholism and scholarships. As alcoholism increased, scholarships decreased.

# <a id='conclusions'></a>
# ## Conclusions
# 
# > **Tip**: Finally, summarize your findings and the results that have been performed. Make sure that you are clear with regards to the limitations of your exploration. If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# > **Tip**: Once you are satisfied with your work, you should save a copy of the report in HTML or PDF form via the **File** > **Download as** submenu. Before exporting your report, check over it to make sure that the flow of the report is complete. You should probably remove all of the "Tip" quotes like this one so that the presentation is as tidy as possible. Congratulations!

# 1-79% of the patients are commited to their appointment.
# 
# 2-The age is the most important indicator to identify if the patient will attend to the appointment.The analysis findings show that the average age of patients that are commited with the appointment is 37 , and the average age of patients who are not showing up is 34.
# 
# 3-We can not depend on the SMS analysis data to identify if the patient will attend to the appointment or not.
# 
# 4-As per our data patients with scholarship are most likely to not showing up.
# 
# 5-the features such as different gender or alcoholic is not a factor to decide if the person would come to his appointment or not!

# ### Limitations:
# Some of the collected data are not logic. The age value can not be 0 or less or more than 100 years.
# 
# The patients collected data are missing some important information that can help in analyzing reasons of no show for patients such as insurance, employee or not, social status(married or single).

# In[34]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




