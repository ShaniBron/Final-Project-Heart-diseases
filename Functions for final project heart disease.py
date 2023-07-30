#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def create_pie_chart(df, column):
    # Create a pie chart using Pandas
    plt.figure(figsize=(5, 5))  # Adjust the figure size as needed
    df[column].value_counts().plot(kind='pie',autopct='%1.1f%%', startangle=140)

    # Add a title
    plt.title(f'Pie Chart {column}')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Show the plot
    plt.show()


# In[3]:


def plot_side_by_side(dataframe,columns, titles=None, figsize=(10, 5)):

    num_plots = len(columns)

    plt.figure(figsize=figsize)

    for i, col in enumerate(columns, 1):
        create_pie_chart(dataframe,col)

    plt.tight_layout()
    plt.show()


# In[4]:


def normalize_plot(df,col_x,col_hue):

    df1 = df.groupby(col_x)[col_hue].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('percent').reset_index()

    g = sns.catplot(x=col_x,y='percent',hue=col_hue,kind='bar',data=df1)
    g.ax.set_ylim(0,100)

    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) + '%'
        txt_x = p.get_x() 
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)


# In[ ]:


def graph_with_lines(df,byparam,param1,param2=None):

    plt.figure(figsize=(8,5))

    df.groupby(byparam).size().plot(kind='bar')

    df.groupby(byparam)[param1].mean().plot(kind='line', secondary_y=True, color='b', label=param1)
    
    if param2==None:
        None
    else:
        df.groupby(byparam)[param2].mean().plot(kind='line', secondary_y=True, color='r', label=param2)

    plt.title(f'{param1} {param2} Average by {byparam}')
    plt.legend(loc="upper left")

