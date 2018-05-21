
# coding: utf-8

# # CSE 255 Programming Assignment 5

# ##  Problem Statement
# 
# In this programming assignment, you will estimate intrinsic dimension by calculating the Mean Squared Distance of the entire dataset to their representative centers. You will use the K-Means API in spark to find representative centers.
# All of the necessary helper code is included in this notebook. However, we advise you to go over the lecture material, the EdX videos and the corresponding notebooks before you attempt this Programming Assignment.

# ## Reviewing the Theory 

# ### Computing the intrinsic dimension of a data set
# Recall from class that given any $d$ dimensional dataset, we can divide it into $n$ cells of diameter $\epsilon$ each. The relationship between $n, d, \epsilon$ is then given by:
# $$
# n = \frac{C}{\epsilon^d}
# $$
# Where $C \in I\!R$
# 
# Alternately, we may write this as:
# $$
# \log{n} = \log{C} + d \times \log{\frac{1}{\epsilon}}
# $$
# 
# Given this expression, we can then compute the dimensionality of a dataset using:
# $$
# d = \frac{\log{n_2} - \log{n_1}}{\log{\epsilon_1} - \log{\epsilon_2}}
# $$
# 
# 
# 
# Where $(n_1,\epsilon_1)$, $(n_2, \epsilon_2)$ are the number of cells and diameter of each cell at 2 different scales.

# ### Using K-Means to estimate intrinsic dimension
# We can use K-Means to approximate the value of intrinsic dimension of a data-set. In this case, the K centers represent the cells, each with a diameter equal to the Mean Squared Distance of the entire dataset to their representative centers. The estimate for intrinsic dimension then becomes:
# $$
# d = \frac{\log{n_2} - \log{n_1}}{\log{\sqrt{\epsilon_1}} - \log{\sqrt{\epsilon_2}}} = 2 \times \frac{\log{n_2} - \log{n_1}}{\log{\epsilon_1} - \log{\epsilon_2}}
# $$

# ## Notebook Setup 

# In[1]:


from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import *
from math import log
import pickle


# In[2]:


spark = SparkSession     .builder     .getOrCreate()
sc = spark.sparkContext


# ## Testing on the homework server
# On EdX, we provide a test submission and a final submission box. The test submission evaluates your code for the test dataset provided with the homework. The final submission is evaluated on a dataset that is much larger.
# 
# For submission and for local testing, make sure to read the path of the file you want to operate with from `./hw5-files.txt`. Otherwise your program will receive no points. This step is exactly the same procedure as your HW2 (Twitter).
# 
# ## Local test
# 
# For local testing, please create your own `hw5-files.txt` file, which contains a single file path on the local disk, e.g.
# `file://<absolute_path_to_current_directory>/hw5-small.parquet`. For final submission, we will create this file on our server for testing with the appropriate file path. If your implementation is correct, you should not worry about which file system (i.e. local file system or HDFS) Spark will read data from.

# In[4]:


with open('./hw5-files.txt') as f:
    file_path =  f.read().strip()


# ## Exercises

# ### Exercise 1: runKmeans

# #### Example
# The function <font color="blue">runKmeans</font> takes as input the complete dataset rdd, the sample of the rdd on which k-means needs to be run on, the k value and the count of elements in the complete data-set. It outputs the Mean Squared Distance(MSD) of all the points in the dataset from their closest centers, where the centers are calculated using k-means.
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# rdd = sc.parallelize([(1,1),(2,2),(3,3),(4,4),(5,5)])
# runKmeans(rdd, sc.parallelize([(1,1),(2,2),(3,3)]), 3, rdd.count())
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# 2.0
# ```
# 
# <font color="red">**Hint : **</font> You might find [K-Means API](https://spark.apache.org/docs/2.2.0/mllib-clustering.html#k-means) useful. Ensure that the initializationMode parameter is set to kmeans++. The computeCost function gives you the sum of squared distances. You might want to tweak maxIterations to compute centers faster

# In[95]:


from math import sqrt

def error(point, clusters):
    center = clusters.centers[clusters.predict(point)]
    return sum([x**2 for x in (point - center)])


def runKmeans(data, sample_dataset, k, count):
    # YOUR CODE HERE
    
    clusters = KMeans.train(sample_dataset, k, maxIterations=10, initializationMode="k-means||")
    MSE = data.map(lambda point: error(point, clusters)).reduce(lambda x, y: x + y)
    
    return MSE / count


# ### Exercise 2: computeIntrinsicDimension

# #### Example
# The function <font color="blue">computeIntrinsicDimension</font> takes as input a pair of values $(n1, e1)$, $(n2, e2)$ and computes the intrinsic dimension as output. $e1, e2$ are the mean squared distances of data-points from their closest center
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# n1 = 10
# n2 = 100
# e1 = 10000
# e2 = 100
# computeIntrinsicDimension(n1, e1, n2, e2)
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# 1.0
# ```
# <font color="red">**Hint : **</font> Use the last formula in the theory section 

# In[24]:


from math import log10
def computeIntrinsicDimension(n1, e1, n2, e2):
    
    d = 2 * (log10(n2) - log10(n1)) / (log10(e1) - log10(e2))
    return d


# ### Exercise 3: Putting it all together

# #### Example
# Now we run K-means for various values of k and use these to estimate the intrinsic dimension of the dataset. Since the dataset might be very large, running kmeans on the entire dataset to find k representative centers may take a very long time. To overcome this, we sample a subset of points from the complete dataset and run Kmeans only on these subsets. We will run Kmeans on 2 different subset sizes: 10000, 20000 points. We will be estimating the MSD for K values of 10, 200, 700, 2000.
# 
# 
# The function <font color="blue">run</font> takes a dataframe containing the complete data-set as input and needs to do the following:
# * For each sample size S
#  * Take the first S elements from the dataframe
#  * For each value of K (number of centroids)
#   * Call runKmeans(data,S,K,data_count)
# * Use the MSD values generated to calculate the intrinsic dimension where $(n_1, n_2) \in \{(10,200),(200,700),(700,2000)\}$.  
# 
# **NOTE: Ensure you the format of your output is identical to the one given below, i.e. the keys have to be of the format:**
# ```python
# ID_<Subset_size>_<K-Value-1>_<K-Value-2>
# ```
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# 
# df = spark.read.parquet(file_path)
# run(df)
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# {'ID_10000_10_200': 1.5574966096390015, 'ID_10000_200_700': 1.3064513902343675, 'ID_10000_700_2000': 1.091310378488035, 'ID_20000_10_200': 1.518279780870003, 'ID_20000_200_700': 1.2660237819996782, 'ID_20000_700_2000': 1.0151131917703071}
# ```
# **Note: The output here is the output of the below function, i.e., the value stored in the variable where the 'run' function is called**

# In[98]:


def run(df):
    S = [10000, 20000]
#     S = [10000]
    K = [10, 200, 700, 2000]
#     K = [10, 200, 700]
    
    rdd = df.rdd.map(lambda x: tuple(float(y) for y in x))
    
    cnt = rdd.count()
    
    res = []
    ret = {}
    for s in S:
        sample_dataset = sc.parallelize(rdd.take(s))
#         sample_dataset = sc.parallelize([tuple(float(x) for x in y) for y in df.head(s)])
        for k in K:
            mse = runKmeans(rdd, sample_dataset, k, s)
            res.append(mse)
            
            
    ret['ID_10000_10_200'] = computeIntrinsicDimension(10, res[0], 200, res[1])
    ret['ID_10000_200_700'] = computeIntrinsicDimension(200, res[1], 700, res[2])
    ret['ID_10000_700_2000'] = computeIntrinsicDimension(700, res[2], 2000, res[3])
    ret['ID_20000_10_200'] = computeIntrinsicDimension(10, res[4], 200, res[5])
    ret['ID_20000_200_700'] = computeIntrinsicDimension(200, res[5], 700, res[6])
    ret['ID_20000_700_2000'] = computeIntrinsicDimension(700, res[6], 2000, res[7])
    
    return ret


# In[99]:


df = spark.read.parquet(file_path)
res = run(df)
with open('results.pkl', 'wb') as output:
    pickle.dump(res, output, 2, fix_imports=True)


# In[100]:


# res

