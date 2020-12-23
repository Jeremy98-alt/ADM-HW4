from collections import defaultdict

import scipy.integrate as integrate
import scipy.special as special

import numpy as np
import pandas as pd
import math
import re

import matplotlib.pyplot as plt
import seaborn as sns

# for the subquestion number 3.1
#from wordcloud import WordCloud 
import functools
import operator
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import cProfile, pstats, io, string
import math
import pickle
import re

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


def groupby_productid_df(df):
    productid_df = pd.DataFrame()
    product_id = []
    reviews = []
    new_df = pd.DataFrame()
    for product, group in df.groupby('ProductId'):
        product_id.append(product)
        reviews.append(" ".join(list(group['Text'])))

    productid_df['ProductId'] = product_id
    productid_df['reviews'] = reviews
    
    return productid_df


def clean_text(text):
    x = re.compile('<.*?>')
    text = re.sub(x, '', text)
    
    stop_words = set(stopwords.words('english')) # obtain the stop words
    good_words = [] # save the correct words to consider like tokens
    tokenizer = RegexpTokenizer("[\w']+") # function to recognize the tokens
    words = tokenizer.tokenize(text) # tokenize the text 
    for word in words:
        # check if the word is lower and it isn't a stop word or a number
        if word.lower() not in stop_words and word.isalpha(): 
            word = PorterStemmer().stem(word) # use the stemmer function
            good_words.append(word.lower()) # insert the good token to lower case
        
    return good_words


class my_Kmeans():

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.prev_labels = [1]
        self.labels = []
        
        
    def initialize_algo(self, matrix):
        random_indices = np.random.choice(len(matrix), size= self.n_clusters, replace=False)
        self.centroids = matrix[random_indices, :]



    def stop_iteration_flag(self):
        if self.labels == self.prev_labels:
            return True
        else:
            return False

    
    def compute_distance(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    
    def assign_clusters(self, matrix):
        self.clusters = {}
        self.prev_labels = self.labels.copy()
        self.labels = []
        
        for row in matrix:
            centroid_idx = np.argmin([self.compute_distance(row, centroid) for centroid in self.centroids])
            self.clusters.setdefault(centroid_idx, []).append(row)
            self.labels.append(centroid_idx)

    
    def update_centroids(self):
        self.centroids = [np.mean(i, axis = 0) for i in self.clusters.values()]

        
    def fit(self, matrix):
        self.initialize_algo(matrix)
        iter_count = 0
        while all((not self.stop_iteration_flag(), iter_count < 100)):
            print("iteration no. {0}".format(iter_count))
            self.assign_clusters(matrix)
            self.update_centroids()
            iter_count += 1
        
        return self.labels
    
    
    def inertia(self, matrix):
        sum_distance = 0
        for i in range(len(matrix)):
            sum_distance += (self.compute_distance(matrix[i], self.centroids[self.labels[i]]))**2
        return sum_distance


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