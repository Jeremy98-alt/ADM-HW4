from collections import defaultdict

import scipy.integrate as integrate
import scipy.special as special

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

# for the subquestion number 3.1
from wordcloud import WordCloud 
import functools
import operator

########## QUESTION 1 ##########

# Our hash function
def hash_function(string_):
    n = 2^32 + 15
    
    result = 0
    for char in string_:
        result = result * 31 + ord(char)
    result = format(result % n, '032b')
    return result

def create_registers():
    return defaultdict(lambda :-1)

# creating the buckets
def update_register(string_, registers):
    b = 12
    x = hash_function(string_)
    j = int(str(x)[:b],2)
    if '1' in set(x[b:]):
        rho_w = (x[b:]).index('1')+1
    else:
        rho_w = len(x[b:])
    registers[j] = max(registers[j],rho_w)

# creating the bucket useful to the hyperLogLog
def process_data(registers):
    with open('hash.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            update_register(line.strip(), registers)

# estimate the cardinality
def hyperLogLog(registers):
    b = 12
    m = 2**b
    alpha = (m)*(integrate.quad(lambda u: (math.log((2+u)/(1+u),2))**(m),0,np.infty )[0])

    Z =(sum(2**-registers[j] for j in registers.keys()))**(-1)
    E = (alpha)**(-1)*(m**2)*Z
    return E

# the error of our filter
def error_rate(registers_count):
    return 1.3 / math.sqrt(2**registers_count)

########## QUESTION 2 ##########



########## QUESTION 3 ##########

## 3.1

# add a column named cluster
def addClusterColumn(new_df, ans):
    new_df["cluster"] = ans
    return new_df

def ListTokenPerCluster(new_df):
    reviews = []
    new_dp = pd.DataFrame()
    for cluster, group in new_df.groupby('cluster'):
        reviews.append(group['reviews'].tolist())
        
    new_dp['reviews'] = reviews
    return new_dp

def showWordClouds(new_dp):
    for k in range(len(new_dp)):
        text = functools.reduce(operator.iconcat, new_dp['reviews'][k], [])
        wordcloud = WordCloud().generate(" ".join(text))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{k} Cluster has this wordcloud")
        plt.show()

## 3.2

def numberOfProduct(ans):
    get_idx, counts_per_cluster = np.unique(ans, return_counts=True)
    print("Show the number of products per each cluster: \n")
    for idx, val in enumerate(counts_per_cluster):
        print("The cluster {} has {} products".format(idx, val))

## 3.3

def createDatasetScore(new_df, ans):
    new_df["cluster"] = ans
    gabibbo = pd.merge(new_df[["ProductId","cluster"]], dt[["ProductId","Score"]], on="ProductId")
    return gabibbo

def showPlotScoreDistribution(gabibbo):
    fig, axes = plt.subplots(3, 2, figsize=(20,20))

    sns.barplot(x = "Score", y = "count", data = gabibbo[gabibbo.cluster == 0].groupby([gabibbo.Score]).Score.count().to_frame('count').reset_index(), ax = axes[0, 0], palette = "GnBu")
    sns.barplot(x = "Score", y = "count", data = gabibbo[gabibbo.cluster == 1].groupby([gabibbo.Score]).Score.count().to_frame('count').reset_index(), ax = axes[0, 1], palette = "GnBu")
    sns.barplot(x = "Score", y = "count", data = gabibbo[gabibbo.cluster == 2].groupby([gabibbo.Score]).Score.count().to_frame('count').reset_index(), ax = axes[1, 0], palette = "GnBu")
    sns.barplot(x = "Score", y = "count", data = gabibbo[gabibbo.cluster == 3].groupby([gabibbo.Score]).Score.count().to_frame('count').reset_index(), ax = axes[1, 1], palette = "GnBu")
    sns.barplot(x = "Score", y = "count", data = gabibbo[gabibbo.cluster == 4].groupby([gabibbo.Score]).Score.count().to_frame('count').reset_index(), ax = axes[2, 0], palette = "GnBu")

    fig.delaxes(axes[2,1])    

## 3.4

def usersWritingCluster(new_df, dt):
    merge_dt = pd.merge(new_df[["ProductId", "cluster"]], dt[["ProductId","UserId"]], on="ProductId")
    return merge_dt.groupby(["cluster"]).UserId.nunique()