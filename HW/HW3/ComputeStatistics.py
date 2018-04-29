
# coding: utf-8

# # CSE 255 Programming Assignment 3

# ##  Problem Statement
# 
# In class, the vectors that we tried to analyze had length, or dimension, of 365, corresponding to the number of 
# days in a year. We outsourced the computation of the math into the `lib/computeStatistics.py` file during the class discussions. In this programming assignment, your task is to fill in the function that is required to efficiently calculate the covariance matrix. All of the necessary helper code is included in this notebook. However, we advise you to go over the lecture material, the EdX videos and the corresponding notebooks before you attempt this Programming Assignment.

# ### Computing Covariance Efficiently

# First, we refresh some of the strategy we went over in class to efficiently compute the covariance matrix while also calculating the mean of the set of vectors. You are required to use these strategies effectively in this assignment.

# To perform [Principle component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
# on vectors, we seperated out two steps to this process:
# 
# 1) Computing the covariance matrix: this is a  simple computation.
# 
# 2) Computing the eigenvector decomposition.

# <font size=4>**In this homework, you will fill up a function that is necessary in order to correctly calculate the covariance matrix in an efficient manner.**</font>

# ## Reviewing the Theory 

# ### Computing the covariance matrix
# Suppose that the data vectors are the column vectors denoted $x$ then the covariance matrix is defined to be
# $$
# E(x x^T)-E(x)E(x)^T
# $$
# 
# Where $x x^T$ is the **outer product** of $x$ with itself.

# If the data that we have is $x_1,x_2,x_n$ then  we estimate the covariance matrix:
# $$
# \hat{E}(x x^T)-\hat{E}(x)\hat{E}(x)^T
# $$
# 
# the estimates we use are:
# $$
# \hat{E}(x x^T) = \frac{1}{n} \sum_{i=1}^n x_i x_i^T,\;\;\;\;\;
# \hat{E}(x) = \frac{1}{n} \sum_{i=1}^n x_i
# $$

# ### Covariance matrix while taking care of `nan`s
# <a id='compCovariance'></a>
# 

# #### The effect of  `nan`s in arithmetic operations  
# * We use an RDD of numpy arrays, instead of Dataframes.
# * Why? Because unlike dataframes, `numpy.nanmean` treats `nan` entries correctly.

# #### Calculating the mean of a vector with nan's 
# * We often get vectors $x$ in which some, but not all, of the entries are `nan`. 
# * We want to compute the mean of the elements of $x$. 
# * If we use `np.mean` we will get the result `nan`. 
# * A useful alternative is to use `np.nanmean` which removes the `nan` elements and takes the mean of the rest.

# #### Computing the covariance  when there are `nan`s
# The covariance is a mean of outer products.
# 
# We calculate two matrices:
# * $S$ - the sum of the matrices, where `nan`->0
# * $N$ - the number of not-`nan` element for each matrix location.
# 
# We then calculate the mean as $S/N$ (division is done element-wise)

# ## Notebook Setup 

# In[1]:


import numpy as np
from numpy import linalg as LA


# In[2]:


# Use this cell to add in extra information that your local setup needs to locate spark.
# import findspark
# findspark.init()

# While submitting, ensure that you comment out this cell


# In[3]:


from pyspark import SparkContext,SparkConf

def create_sc(pyFiles):
    sc_conf = SparkConf()
    sc = SparkContext(pyFiles=pyFiles)
    
    return sc 

sc = create_sc(pyFiles=['lib/numpy_pack.py','lib/computeStatistics.py'])


# # Computing Statistics 

# ## Computing the mean together with the covariance
# <a id='compCovariances'></a>
# To compute the covariance matrix we need to compute both $\hat{E}(x x^T)$ and $\hat{E}(x)$. Using a simple trick, we can compute both at the same time.

# Here is the trick: lets denote a $d$ dimensional **column vector** by $\vec{x} = (x_1,x_2,\ldots,x_d)$ (note that the subscript here is the index of the coordinate, not the index of the example in the training set as used above). 
# 
# The augmented vector $\vec{x}'$ is defined to be the $d+1$ dimensional vector $\vec{x}' = (1,x_1,x_2,\ldots,x_d)$.

# The outer product of $\vec{x}'$ with itself is equal to 
# 
# $$ \vec{x}' {\vec{x}'}^T
# = \left[\begin{array}{c|ccc}
#     1 &  &{\vec{x}}^T &\\
#     \hline \\
#     \vec{x} & &\vec{x} {\vec{x}}^T \\ \\
#     \end{array}
#     \right]
# $$
# 
# Where the lower left matrix is the original outer product $\vec{x} {\vec{x}}^T$ and the first row and the first column are $\vec{x}^T$ and $\vec{x}$ respectively.

# Now suppose that we take the average of the outer product of the augmented vector and convince yourself that:
# $$
# \hat{E}(\vec{x}' {\vec{x}'}^T) = \frac{1}{n} \sum_{i=1}^n {\vec{x}'}_i {\vec{x}'}_i^T
# = \left[\begin{array}{c|ccc}
#     1 &  &\hat{E}(\vec{x})^T &\\
#     \hline \\
#     \hat{E}(\vec{x}) & &\hat{E}(\vec{x} {\vec{x}}^T) \\ \\
#     \end{array}
#     \right]
# $$
# 
# So indeed, we have produced the outer product average together with (two copies of) the average $\hat{E}(\vec{x})$

# ## Helper Functions

# ### OuterProduct

# #### Description
# The function <font color="blue">outerProduct</font> computes outer product and indicates which locations in matrix are undefined.
# 
# **Input**: X is a 1 x n matrix 
# 
# **Output**: The output is a tuple of:
# 1. O is a n x n matrix which is the outer product of the two matrices.
# 2. N is a n x n matrix which represents whether each position in the matrix has a "valid" non-NaN element.

# In[4]:


def outerProduct(X):
    O=np.outer(X,X)
    N=1-np.isnan(O)
    return (O,N)


# ### sumWithNan

# #### Description
# 
# The function <font color="blue">sumWithNan</font> adds two pairs of (**matrix**, **count**) where **matrix** and **count** are the O and N returned from the outerProduct function.
# 
# **Input** : M1 and M2 are tuples of n x n matrices. The first matrix in each tuple is derived from the
#     outer product and the second matrix in each tuple represents the count of non-NaN elements in that position
# 
# **Output** : Two matrices. The first (X) contains the Nansum of elements in the outer-product matrix in M1 and M2 and the second (N) contains the count of non-Nan elements in M1 and M2. This output has the same shape as the input i.e a tuple of n x n matrices.

# In[5]:


def sumWithNan(M1,M2):
    (X1,N1)=M1
    (X2,N2)=M2
    N=N1+N2
    X=np.nansum(np.dstack((X1,X2)),axis=2)
    return (X,N)


# # Exercise

# ## Description

# The function <font color="blue">HW_func</font> takes in two $n$ x $n$ matrices, S and N.
# 
# The first $n$ x $n$ matrix, `S`, is the output from reducing the outer product of vectors by taking the sum at each position in the outer product. Remember from the theory that the vectors have been augmented with a leading 1 to facilitate the computation of the mean and the co-variance in the same computation.
# 
# The second $n$ x $n$ matrix, `N`, is derived from reducing boolean matrices that denote the presence of a valid value in the outer product of a vector. The reduction is done by summing up the boolean matrices. This means that the $n$ x $n$ matrix would contain the count of valid not-nan entries at each position in the outer product. 
# 
# For example, if the vectors that we want to do PCA on are:
# 
# `[array([-0.09993104,         nan]), array([-0.17819987, -0.70368551])]`
# 
# Then the matrix `S` would be:
# 
# `[[ 2.         -0.2781309  -0.70368551]
#  [-0.2781309   0.0417414   0.12539666]
#  [-0.70368551  0.12539666  0.4951733 ]]`
# 
# 
#  And the matrix `N` would be:
#  
#  `[[2 2 1]
#  [2 2 1]
#  [1 1 1]]`
#  
#  
#  Note how `S` and `N` are generated:
#  
#  ```
#  x = np.array([1, -0.09993104, np.nan]
#  y = np.array([1, -0.17819987, -0.70368551])
#  S,N = sumWithNan(outerProduct(x),outerProduct(y))
#  ```
#  
#  
#  The matrices `S` and `N` are both `numpy.ndarrays`
# 
# You have to calculate and return the following statistics: 
# 
# 1. E : The nan-sum of the vectors, as described in [Computing Covariance With NaNs](#compCovariance)
# 
# 2. NE : The number of not-nan entries for each coordinate of the vectors
# 
# 3. Mean : The Mean vector (ignoring nans)
# 
# 4. O :  The sum of the outer products
# 
# 5. NO : The number of non-nans in the outer product.
# 
# Be careful with the data types of variables returned from ```HW_func()```
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# my_S = np.array([[1, 2, 3],[2,4,5][3,5,6]])
# my_N = np.array([[2, 2, 1],[2,2,1],[1,1,1]])
# 
# HW_func(my_S, my_N)
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# E = np.array([2, 3])
# NE = np.array([ 2.,  1.])
# Mean = np.array([ 1.,  3.])
# O = np.array([[4, 5],
#                [5, 6]])
# NO = array([[ 2.,  1.],
#            [ 1.,  1.]])
# ```

# ## Definition

# In[6]:


def HW_func(S,N):
    # E is the sum of the vectors
    # NE is the number of not-nan entries for each coordinate of the vectors
    # Mean is the Mean vector (ignoring nans)
    # O is the sum of the outer products
    # NO is the number of non-nans in the outer product.
    
    ## NOTE: All of these computations require just a single line of code each. However, be careful with slicing of indexes.
    
    # YOUR CODE HERE
    E = S[0, 1:]
    NE = N[0, 1:].astype(np.float64)
    Mean = E / NE
    O = S[1:, 1:]
    NO = N[1:, 1:].astype(np.float64)
    return E, NE, Mean, O, NO
    


# ## Tests

# ### Test 1

# In[7]:


S = np.array([[ 2.0, 0.24553034, -0.03128947], [0.24553034, 0.06099381, -0.38770712], [-0.03128947, -0.38770712, 4.77673193]])
N = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])

E, NE, Mean, O, NO = HW_func(S, N)

expected_result = {'E': np.array([ 0.24553034, -0.03128947]), 'NE': np.array([ 2.,  2.]), 
          'O': np.array([[ 0.06099381, -0.38770712], [-0.38770712,  4.77673193]]), 
          'NO': np.array([[ 2.,  2.],[ 2.,  2.]]), 'Mean': np.array([ 0.12276517, -0.01564473]) 
         }


# #### Type Check

# In[8]:


assert type(E) == np.ndarray, "Incorrect return type. Should return np.array"


# In[9]:


assert type(NE) == np.ndarray, 'Invalid return type. We expected numpy.ndarray'
assert type(NE[0]) == np.float64, 'Invalid return type. Each element in numpy.ndarray should                     be numpy.float64'


# In[10]:


assert type(Mean) == np.ndarray, "Incorrect return type. Should return np.array"


# In[11]:


assert type(O) == np.ndarray, "Incorrect return type. Should return np.array"


# In[12]:


assert type(NO) == np.ndarray, 'Invalid return type. We expected numpy.ndarray'
assert type(NO[0,0]) == np.float64, 'Invalid return type. Each element in numpy.ndarray should                     be numpy.float64. Your elements have datatype ' + str(type(NO[0,0]))


# #### Shape Check

# In[13]:


assert E.shape == (2,), "Returned np.array should be a vector of size n-1 = 2"


# In[14]:


assert NE.shape == (2,), "Returned np.array should be a vector of size n-1 = 2"


# In[15]:


assert Mean.shape == (2,), "Returned np.array should be a vector of size n-1 = 2"


# In[16]:


assert O.shape == (2,2), "We expected a numpy ndarray of shape (2,2), You returned: " + str(O.shape)


# In[17]:


assert NO.shape == (2,2), "We expected a numpy ndarray of shape (2,2), You returned: " + str(NO.shape)


# #### Value Check

# In[18]:


assert (np.around(E, decimals=6) == np.around(expected_result['E'], decimals=6)).all(), "Output value of E does not match expected output of function. You returned " + str(E)


# In[19]:


assert (np.around(NE, decimals=6) == np.around(expected_result['NE'], decimals=6)).all(), "Output value of NE does not match expected output of function. You returned " + str(NE)


# In[20]:


assert (np.around(Mean, decimals=6) == np.around(expected_result['Mean'], decimals=6)).all(), "Output value of Mean does not match expected output of function. You returned " + str(Mean)


# In[21]:


assert (np.around(O, decimals=6) == np.around(expected_result['O'], decimals=6)).all(), "Output value of O does not match expected output of function. You returned " + str(O)


# In[22]:


assert (np.around(NO, decimals=6) == np.around(expected_result['NO'], decimals=6)).all(), "Output value of NO does not match expected output of function. You returned " + str(NO)


# ### Test 2

# In[23]:


S = np.array([[ 2., -0.92050828, -0.90843676], [-0.92050828, 0.51012277, 0.60698693], [-0.90843676, 0.60698693, 0.82525735]])
N = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1]])
E, NE, Mean, O, NO = HW_func(S, N)

expected_result = {'E': np.array([-0.92050828, -0.90843676]), 'NE': np.array([ 2.,  1.]), 
          'O': np.array([[0.51012277,  0.60698693], [0.60698693,  0.82525735]]), 
          'NO': np.array([[ 2.,  1.], [ 1.,  1.]]), 'Mean': np.array([-0.46025414, -0.90843676]) 
         }


# In[24]:


assert (np.around(E, decimals=6) == np.around(expected_result['E'], decimals=6)).all(), "Output value of E does not match expected output of function. You returned " + str(E)


# In[25]:


assert (np.around(NE, decimals=6) == np.around(expected_result['NE'], decimals=6)).all(), "Output value of NE does not match expected output of function. You returned " + str(NE)


# In[26]:


assert (np.around(Mean, decimals=6) == np.around(expected_result['Mean'], decimals=6)).all(), "Output value of Mean does not match expected output of function. You returned " + str(Mean)


# In[27]:


assert (np.around(O, decimals=6) == np.around(expected_result['O'], decimals=6)).all(), "Output value of O does not match expected output of function. You returned " + str(O)


# In[28]:


assert (np.around(NO, decimals=6) == np.around(expected_result['NO'], decimals=6)).all(), "Output value of NO does not match expected output of function. You returned " + str(NO)


# ### Hidden test 1

# In[29]:


# Hidden Tests here


# In[30]:


# Hidden Tests here


# In[31]:


# Hidden Tests here


# In[32]:


# Hidden Tests here


# In[33]:


# Hidden Tests here


# ### Hidden test 2

# In[34]:


# Hidden Tests here


# In[35]:


# Hidden Tests here


# In[36]:


# Hidden Tests here


# In[37]:


# Hidden Tests here


# In[38]:


# Hidden Tests here


# # Covariance, Eigen values and Eigen Vectors

# ## computeCov

# ### Description
# 
# The function <font color="blue">computeCov</font> calls the <font color="blue">HW_func</font> and uses the values returned to compute covariance.
# 
# **Input**: RDD containing a set of numpy arrays (vectors), all of the same length.
# 
# **Output**: This returns a dictionary containing the E, NE, O, NO and Mean computed by the <font color="blue">HW_func</font> along with the variance(Var) and covariance(Cov) matrix computed for that set of vectors.
# 
# You are not expected to change this function. This is only for understanding how the values computed in the <font color="blue">HW_func</font> contribute to the computation of the covariance matrix.

# In[39]:


def computeCov(RDDin):
    """
    computeCov receives as input an RDD of np arrays, all of the same length, 
    and computes the covariance matrix for that set of vectors
    """
    RDD=RDDin.map(lambda v:np.array(np.insert(v,0,1),dtype=np.float64))
    # insert a 1 at the beginning of each vector so that the same 
    # calculation also yields the mean vector
    OuterRDD=RDD.map(outerProduct)   # separating map and reduce does not matter because of Spark's lazy execution
    (S,N)=OuterRDD.reduce(sumWithNan)
    E,NE,Mean,O,NO=HW_func(S,N)
    Cov=O/NO - np.outer(Mean,Mean)
    # Output also the diagnal which is the variance for each day
    Var=np.array([Cov[i,i] for i in range(Cov.shape[0])])
    return {'E':E,'NE':NE,'O':O,'NO':NO,'Cov':Cov,'Mean':Mean,'Var':Var}


# ## The process function

# ### Description
# 
# The function <font color="blue">process</font> calls the <font color="blue">computeCov</font> and uses the covariance matrix returned to compute the Eigen Values and Eigen Vectors.
# 
# **Input**: A list of numpy arrays (vectors), all of the same length.
# 
# **Output**: This returns the Eigen value and Eigen Vector matrix for the given set of vectors.
# 
# You are not expected to change this function. This is only for understanding how the values computed in the <font color="blue">HW_func</font> contribute to the computation of the covariance matrix and consequently the Eigen Values and Eigen Vectors.

# In[40]:


def process(data_list):
    # compute covariance matrix
    RDD=sc.parallelize(data_list)
    OUT=computeCov(RDD)
    #find PCA decomposition
    eigval,eigvec=LA.eig(OUT['Cov'])
    return eigval, eigvec


# ### Tests

# In[41]:


data_list = ([np.array([ -1.43475066e-03,   1.52970999e+00]), np.array([ 0.24696509, -1.56099945])])
eigval, eigvec = process(data_list)
expected_result = {'eigval': np.array([0., 2.40354683]), 
                   'eigvec': np.array([[-0.99678591, 0.08011153], [-0.08011153, -0.99678591]])
                  }

assert (np.around(eigval, decimals=6) == np.around(expected_result['eigval'], decimals=6)).all(), "Output value of eigval does not match expected output of function. You returned " + str(eigval)

assert (np.around(eigvec, decimals=6) == np.around(expected_result['eigvec'], decimals=6)).all(), "Output value of eigvec does not match expected output of function. You returned " + str(eigvec)


# In[42]:


data_list = ([np.array([-0.25234187,         np.nan]), np.array([-0.66816641, -0.90843676])])
eigval, eigvec = process(data_list)
expected_result = {'eigval': np.array([ 0.21172155, -0.16849404]), 
                   'eigvec': np.array([[ 0.74622118, -0.66569809], [ 0.66569809,  0.74622118]])
                  }
assert (np.around(eigval, decimals=6) == np.around(expected_result['eigval'], decimals=6)).all(), "Output value of eigval does not match expected output of function. You returned " + str(eigval)

assert (np.around(eigvec, decimals=6) == np.around(expected_result['eigvec'], decimals=6)).all(), "Output value of eigvec does not match expected output of function. You returned " + str(eigvec)


# In[43]:


#Hidden Tests here


# In[44]:


#Hidden Tests here


# # Submission Instructions
# 
# 1. Validate your submission by submitting your assignment on **"Homework 3 Test Submission"** under **"Week 3"** on edX. This is worth 10 points.
# 2. Submit the final copy of your assignment on **"Homework 3 Final Submission"** under **"Week 3"** on edX. This will be evaluated for visible and hidden tests and is worth 90 points.
# 
