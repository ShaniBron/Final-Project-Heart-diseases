# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report,\
                             roc_auc_score, roc_curve, recall_score, precision_score, average_precision_score

from sklearn.preprocessing import LabelEncoder


mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']

def create_pie_chart(df, column, figsize=(3, 3)):

    plt.figure(figsize=figsize)
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

def normalize_plot(df,col_x,col_hue, order_x=None, height=4, aspect = 2):

    df1 = df.groupby(col_x)[col_hue].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('percent').reset_index()

    g = sns.catplot(x=col_x,y='percent',hue=col_hue,kind='bar',data=df1, order=order_x, height=height, aspect=aspect, palette=mypal[1::4])

    g.ax.set_ylim(0,100)

    for p in g.ax.patches:
        txt = str(p.get_height().round(2)) + '%'
        txt_x = p.get_x() 
        txt_y = p.get_height()
        g.ax.text(txt_x,txt_y,txt)

    g.set(title=f"Percentage of {col_x} by {col_hue}")

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

def report(clf, X, y):
    pred = clf.predict(X)
    predproba = clf.predict_proba(X)[:, 1]
    cm = pd.DataFrame(confusion_matrix(y_true=y, 
                                       y_pred=pred), 
                      index=clf.classes_, 
                      columns=clf.classes_)
    precision = precision_score(y,pred)
    recall = recall_score(y,pred)
                 
    auc = roc_auc_score(y, predproba)
    
    PR_curve = average_precision_score(y,predproba)

    report_dict = {'cm': cm, 'precision': precision, 'recall': recall, 'AUC': auc, 'PR': PR_curve}
    return report_dict
  

def printreport(clf, X, y, Xtest=None, ytest=None):
    
    train = report(clf, X, y)
    
    if Xtest is not None and ytest is not None:
        test = report(clf, Xtest, ytest)

        return print('Train Confusion Matrix:\n{}\n\nTest Confusion Matrix:\n{}\n\nTrain Precision: {:.3f}\nTest Precision: {:.3f}\n\nTrain Recall: {:.3f}\nTest Recall: {:.3f}\n\nTrain ROC AUC: {:.3f}\nTest ROC AUC: {:.3f}\n\nTrain PR Curve: {:.3f}\nTest PR Curve: {:.3f}'.\
            format(train['cm'], test['cm'], train['precision'], test['precision'], train['recall'], test['recall'], train['AUC'], test['AUC'], train['PR'], test['PR']))

    else:    
        return print('Confusion Matrix:\n{}\n\nPrecision: {:.3f}\nRecall: {:.3f}\nROC AUC: {:.3f}\nPR Curve: {:.3f}'.format(train['cm'], train['precision'], train['recall'], train['AUC'], train['PR']))
    

def transform(X,list_col_indx,list_col_rplce=None,list_what_rplce=None):
        cat_cols = LabelEncoder()
        X_copy=X.copy()
        X_copy[list_col_indx] = X_copy[list_col_indx].apply(lambda col:cat_cols.fit_transform(col))
        
        if list_col_rplce is not None and list_what_rplce is not None:
            for i in range(len(list_col_rplce)):
                for key, value in list_what_rplce[i].items():
                    X_copy[list_col_rplce[i]].replace(key, value, inplace=True)

        return X_copy


def graph_distrib(df,distrb_col,by_col):
    
    fig, ax = plt.subplots(figsize = (13,5))

    sns.kdeplot(df[df[by_col]==df[by_col].unique()[1]][distrb_col], alpha=0.5,shade = True, color="red", label=f"{by_col} {df[by_col].unique()[1]}", ax = ax)
    sns.kdeplot(df[df[by_col]==df[by_col].unique()[0]][distrb_col], alpha=0.5,shade = True, color="#fccc79", label=f"{by_col} {df[by_col].unique()[0]}", ax = ax)
    plt.title(f'Distribution of {distrb_col} by {by_col}', fontsize = 18)
    ax.set_xlabel(f"{distrb_col}")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()


def result_table(list_of_models,list_of_models_names,X,y):
    results_table = pd.DataFrame(columns=['Model','Recall', 'Precision', 'AUC', 'PR'])
    for i in range(len(list_of_models)):
        pred = list_of_models[i].predict(X)
        predproba = list_of_models[i].predict_proba(X)[:, 1]

        dict_m ={'Model':list_of_models_names[i], 'Recall': recall_score(y_test,pred), 'Precision': precision_score(y,pred), \
                'AUC': roc_auc_score(y, predproba),'PR': average_precision_score(y,predproba)}

        results_table = results_table.append(dict_m,ignore_index=True)
    
    results_table.set_index('Model',inplace=True)

    return results_table

def roc_curve_plot(list_of_models,list_of_models_names,X,y):
    plt.figure(figsize=(10,8))
    for i in range(len(list_of_models)):
        y_pred_proba = list_of_models[i].predict_proba(X)[::,1]
        fpr, tpr, _ = roc_curve(y,  y_pred_proba)
        plt.plot(fpr,tpr,label=list_of_models_names[i])
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()