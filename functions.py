import matplotlib.pyplot as plt
import seaborn as sns

def create_pie_chart(df, column):

    plt.figure(figsize=(5, 5))
    df[column].value_counts().plot(kind='pie',autopct='%1.1f%%', startangle=140)

    plt.title(f'Pie Chart {column}')

    plt.axis('equal')

    plt.show()


def plot_side_by_side(df,columns, titles=None, figsize=(10, 5)):

    num_plots = len(columns)

    plt.figure(figsize=figsize)

    for i, col in enumerate(columns, 1):
        create_pie_chart(df,col)

    plt.tight_layout()
    plt.show()

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
