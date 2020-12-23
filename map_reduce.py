import numpy as np

def map_(centroids,x_i):
    #centroids = set of cluster centers
    cluster_i = np.argmin([np.linalg.norm(mu_j-np.array(x_i)) for mu_j in centroids])
    
    #cluster_i = cluster label, x_i = datapoint
    return cluster_i

def reduce(x_incluster_j):
    return list(np.mean(list(x_incluster_j),axis = 0))

def arr(x):
    return list(map(float, x.split(","))) 

def k_means_mapreduce(matrix,k,prev_cluster = None,centroids = None):
    if prev_cluster == None:
        centroids = matrix.takeSample(False, k)
        
    cluster = matrix.map(lambda x: ((map_(centroids,x),x)))
    
    if prev_cluster != None: # find per each cluster a point
        print(prev_cluster.collect()) 
        print(cluster.collect())        

    if prev_cluster != None and prev_cluster.keys().collect() == cluster.keys().collect():
        return cluster.keys().collect()
    else:   
        #update of centroids
        upd_centroids = cluster.groupByKey().mapValues(lambda x_incluster_j: reduce(x_incluster_j))
        print('centroids:',upd_centroids.collect())
        return k_means_mapreduce(matrix, k,cluster,centroids = [t[1] for t in upd_centroids.collect()])

if __name__ == "__main__":
    matrix = sc.textFile("./Reviews.csv")
    matrix = matrix.map(lambda x: arr(x))
    k_means_mapreduce(matrix, 10)
