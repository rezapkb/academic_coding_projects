#!/usr/bin/env python
# coding: utf-8

# 
# # We load the dataset and examine the first 5 records.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv('SUSY.csv.gz', header=None)
df.head()


# In[2]:



print(df.values.shape)


# # Histograms
# 
# 

# In[3]:



print(df[df.columns[0:1]].describe())
df[df.columns[0:1]].hist(figsize=(5, 5), bins=5, xlabelsize=8, ylabelsize=8);


# In[4]:


print(df[df.columns[1:19]].describe())
df[df.columns[1:19]].hist(figsize=(20, 20), bins=100, xlabelsize=8, ylabelsize=8);


# # Box plot to identify outliers
# 
# 

# In[5]:


print(df[[3,6,8]].describe())


# In[6]:


import warnings
warnings.filterwarnings("ignore")

for i in [3,6,8, 14]:
  plt.figure(figsize=(4,4))
  sns.boxplot(df[i], orient='V')


# #  correlation of different features

# In[7]:


df_corr = df.corr()[[0]][1:19]
print(df_corr.sort_values(by=0,ascending=False))


# # Scatter Plots of labels vs feature values to identify non-linear relationship

# In[8]:



for i in range(1, len(df.columns), 3):
    sns.pairplot(data=df,
                x_vars=df.columns[i:i+3],
                y_vars=[0])


# #  Correlation Matrix 

# In[9]:



corr = df.drop(0, axis=1).corr() # We already examined Label correlations
plt.figure(figsize=(10, 8))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# # PCA - Reduction of Dimensionality

# In[10]:



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_scaled = sc.fit_transform(df.drop(0))
from sklearn.decomposition import PCA
covar_matrix = PCA(n_components = 18)
covar_matrix.fit(df_scaled)
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print(var)
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)


# # Clustering : Except K-Means and SOM; Other options we tried were too mcuh time-consuming to perform. E.g: Gaussian Mixture & DBSCAN

# # Clustering using K-Means algorithm
# 
# ### Elbow Method to find out optimal value of k

# In[11]:



from sklearn.cluster import KMeans
Error =[]
for i in range(1, 20):
    print(i)
    kmeans = KMeans(n_clusters = i).fit(df.drop(0))
    Error.append(kmeans.inertia_)
plt.plot(range(1, 20), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


# # Clustering using SOM

# In[22]:


from minisom import MiniSom
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv('SUSY.csv.gz', header=None)
data.head()
#data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
#                    names=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
#                   'asymmetry_coefficient', 'length_kernel_groove', 'target'], usecols=[0, 5], 
#                   sep='\t+', engine='python')
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# Initialization and training
som_shape = (1, 3)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)

som.train_batch(data, 500, verbose=True)


# In[23]:


# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=80, linewidths=35, color='k', label='centroid')
plt.legend();


# The number of clusters obtained seems very inconsistent though!

# ## Another Minisom Clustering

# In[6]:


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Minisom library and module is used for performing Self Organizing Maps

from minisom import MiniSom


# In[7]:


# Loading Data

data = pd.read_csv('SUSY.csv.gz')

# X 

data

# Shape of the data:

data.shape

# Info of the data:

data.info()


# In[8]:


# Defining X variables for the input of SOM
X = data.iloc[:, 1:14].values
y = data.iloc[:, -1].values
# X variables:
pd.DataFrame(X)


# In[9]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
pd.DataFrame(X)


# In[10]:


# Set the hyper parameters
som_grid_rows = 100
som_grid_columns = 100
iterations = 20000
sigma = 1
learning_rate = 0.5


# In[11]:


#define SOM:

som = MiniSom(x = som_grid_rows, y = som_grid_columns, input_len=13, sigma=sigma, learning_rate=learning_rate)

# Initializing the weights

som.random_weights_init(X)

# Training

som.train_random(X, iterations)


# In[12]:


from pylab import plot, axis, show, pcolor, colorbar, bone
bone()
pcolor(som.distance_map().T)       # Distance map as background
colorbar()
show()


# Too Time Consuming

# ## Using SUSI module for clustering data

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix
data = pd.read_csv('SUSY.csv.gz')
som = susi.SOMClustering(
    n_rows=30,
    n_columns=30
)
som.fit(data)


# In[5]:


u_matrix = som.get_u_matrix()
plot_umatrix(u_matrix, 30, 30)
plt.show()


# It seems that almost 10 true clusters exists within the data, which is close to the result obtained from k-means

# # CLASSIFICATION

# # After successful clustering we use some algorithms to investigate the SUSY dataset.

# # Preparing the Dataset

# In[57]:


from sklearn.utils import shuffle

X = df.values[:,1:]
y = df.values[:,0]

#we shuffle the dataset as follows:
X, y = shuffle(X, y)


# # Decision Tree with Pruning
# 

# In[58]:



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[:100000], y[:100000], test_size=0.2, random_state=0)


clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print("Training accuracy:", clf.score(X_train, y_train))
print("Testing accuracy:", clf.score(X_test, y_test))


# ## Minimum Cost Complexity Pruning
# 

# In[59]:


path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


# In[60]:



fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")


# In[61]:



clfs = []

ccp_alphas = np.linspace(0.00005, 0.001, 10)
print(ccp_alphas.shape)
for ccp_alpha in ccp_alphas:
    print("Training with value: ", ccp_alpha)
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


# In[62]:



train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[63]:


clf_final = DecisionTreeClassifier(random_state=0, ccp_alpha=1e-4)
clf_final.fit(X_train, y_train)
print("Training accuracy: ", clf_final.score(X_train, y_train))
print("Test accuracy: " , clf_final.score(X_test, y_test))


# # Random Forests Algorithm

# In[64]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[:1000000], y[:1000000], test_size=0.2, random_state=0)

#create a RandomForestClassifier with 5 estimators
clf = RandomForestClassifier(n_estimators=5)
clf.fit(X_train, y_train)

print("Training done")
print("Training accuracy: ", clf.score(X_train, y_train))
print("Test set accuracy: ", clf.score(X_test, y_test))


# In[65]:



val_acc = []
tr_acc = []
for depth in range(2, 20, 2): 
  clf = RandomForestClassifier(n_estimators=10, max_depth = depth)
  clf.fit(X_train, y_train)
  print("For depth = ", str(depth))
  print("Training done")
  print("Training accuracy: ", clf.score(X_train, y_train))
  print("Test set accuracy: ", clf.score(X_test, y_test))
  val_acc.append(clf.score(X_test, y_test))
  tr_acc.append(clf.score(X_train, y_train))
  print()

plt.plot(val_acc, label='Validation accuracy')
plt.plot(tr_acc, label='Training accuracy')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
     


# # Naive Bayes
# 

# In[66]:


from sklearn.naive_bayes import GaussianNB

#we use the entire training data set, with a 20% split for testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Training Accuracy: ", gnb.score(X_train, y_train))
print("Testing Accuracy: ", gnb.score(X_test, y_test))


# # K-Nearest Neighbours

# In[67]:



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.2, random_state=0)

val_acc = []
tr_acc = []

for k in range(10, 30, 2):
  clf = KNeighborsClassifier(k)
  clf.fit(X_train, y_train)
  print("Value of k: ", k)
  print("Training accuracy: ", clf.score(X_train, y_train))
  print("Validation accuracy: ", clf.score(X_test, y_test))
  val_acc.append(clf.score(X_test, y_test))
  tr_acc.append(clf.score(X_train, y_train))
  print()

plt.plot(range(10, 30, 2), val_acc, label='Validation accuracy')
plt.plot(range(10, 30, 2), tr_acc, label='Training accuracy')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# # Bagging with Trees

# In[80]:



from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier 

X_train, X_test, y_train, y_test = train_test_split(X[:100000], y[:100000], test_size=0.2, random_state=0)

val_acc = []
tr_acc = []

for n_estimators in range(10, 50, 10):
  bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.3, n_estimators=n_estimators).fit(X_train,y_train)
  print("Number of estimators ", n_estimators)
  val_acc.append(bagging.score(X_test, y_test))
  tr_acc.append(bagging.score(X_train, y_train))
  print("Training Accuracy: ", bagging.score(X_train, y_train))
  print("Testing Accuracy: ", bagging.score(X_test, y_test))
  


# In[81]:



plt.plot(range(10, 50, 10), val_acc, label='Validation accuracy')
plt.plot(range(10, 50, 10), tr_acc, label='Training accuracy')
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ### Results
# We obtain the following results on each of the different classification techniques we used:
# 
# Decision tree with pruning: 78%
# 
# Random Forests: 79.8%
# 
# k-NN: 77.2%
# 
# Naive Bayes: 73%
# 
# Artificial Neural Networks: 80%
# 
# Bagging with Decision Trees: 79.6%

# # APPENDIX.1

# ## In the follwing we tried some trials to build neural networks with different configurations to figure out if we can do the classification by NN. However, due to the process being too heavy for the system we interrupted the kernels almost after two hours and realized that the process is beyond the capability of my system, the very initial results seems very promising using this method. Thus,  one can expect great results using NNs. Yet, using them is unreasonable somehow because of the energy and Time required for processing.

# # Importing the SUSY data set with Pandas and splitting 90-10 for training and testing.

# In[85]:


# Importing the SUSY Data set
import sys, os
import pandas as pd

import numpy as np
import warnings
#Commnet the next line on to turn off warnings
#warnings.filterwarnings('ignore')


seed=12
np.random.seed(seed)
import tensorflow as tf
# suppress tflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(seed)

# Download the SUSY.csv (about 2GB) from UCI ML archive and save it in the same directory as this jupyter notebook
# See: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/
#filename="SUSY.csv"
path = "~/SUSY.csv.gz"
filename=full_path = os.path.expanduser(path)

columns=["signal", "lepton 1 pT", "lepton 1 eta", "lepton 1 phi", "lepton 2 pT", "lepton 2 eta", 
         "lepton 2 phi", "missing energy magnitude", "missing energy phi", "MET_rel", 
         "axial MET", "M_R", "M_TR_2", "R", "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos(theta_r1)"]

# Load 1,500,000 rows as train data, 50,000 as test data
df_train=pd.read_csv(filename,names=columns,nrows=1500000,engine='python')
df_test=pd.read_csv(filename,names=columns,nrows=50000, skiprows=1500000,engine='python')


# # Run logistic regression using Linear Model functions

# In[86]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
#import ml_style as style #optional styling sheet
#mpl.rcParams.update(style.style) #optional styling sheet

def getTrainData(nVar):
    designMatrix = df_train.iloc[:,1:nVar+1].values
    #now the signal
    labels = df_train['signal'].values # labels (0 or 1)
    return (designMatrix,labels)

def getTestData(nVar):
    designMatrix = df_test.iloc[:,1:nVar+1].values
    #now the signal
    labels = df_test['signal'].values
    return (designMatrix,labels)

# define
def build_roc_curve(probs, signal_bit, threshes):
    # Convert things to a pandas series to build a DataFrame
    # which will make ROC curve logic easier to express
    signal_probs = pd.Series(probs[:,1])
    signal_true = pd.Series(signal_bit)
    signal_df = pd.DataFrame(signal_probs, columns=['sig_prob'])
    signal_df.loc[:,'sig_true'] = signal_true
    Acceptance = []
    Rejection = []
    for thresh in threshes:
        # define acceptance
        signal_df.loc[:,'accept'] = signal_df['sig_prob'] > thresh
        # sum over data frame with slicing conditions
        nSigCor = len(signal_df[(signal_df['accept']) & (signal_df['sig_true']==1.)])
        nSig = len(signal_df[signal_df['sig_true']==1.])
        nBkgCor = len(signal_df[ (signal_df['sig_true']==0.) & (~signal_df['accept'])])
        nBkg = len(signal_df[signal_df['sig_true']==0.])
        Acceptance.append(nSigCor/nSig) # False positive rate
        Rejection.append(nBkgCor/nBkg) # True positive rate

    return Acceptance, Rejection
    
# let's define this as a function so we can call it easily
def runTensorFlowRegression(nVar,alpha):

    #make data array placeholder for just first 8 simple features
    x = tf.placeholder(tf.float32,[None,nVar])
    #make weights and bias
    W = tf.Variable(tf.zeros([nVar,2])) #we will make y 'onehot' 0 bit is bkg, 1 bit is signal
    b = tf.Variable(tf.zeros([2]))

    #make 'answer variable'
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    #placeholder for correct answer
    y_ = tf.placeholder(tf.float32, [None, 2])
    #cross entropy with L2 regularizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)+alpha*tf.nn.l2_loss(W))
    
    #define training step
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #initialize variables 
    init = tf.global_variables_initializer()
    #setup session
    sess = tf.Session()
    sess.run(init)

    #ok now everything is setup for tensorflow, but we need the data in a useful form
    #first let's get the variables
    Var_train, Sig_train_bit1 = getTrainData(nVar)
    #now the signal
    Sig_train_bit0 = Sig_train_bit1.copy()
    Sig_train_bit0 = 1 - Sig_train_bit0
    Sig_train = np.column_stack((Sig_train_bit0,Sig_train_bit1))
    
    ######## ------- TRAINING ----------############
    #Now perform minibatch gradient descent with minibatches of size 100:
    n_data = len(Sig_train_bit1)
    minibatch_size = 1000
    n_minibatch = n_data//minibatch_size
    print('\t Training with %i minibatches, dataset size is %i'%(n_minibatch,n_data))
    for i in range(0, n_minibatch):
        sys.stdout.write("%.3f %% completed \r"%(100*i/n_minibatch))
        sys.stdout.flush()
        start = i*minibatch_size
        end = (i+1)*minibatch_size-1
        
        batch_x = Var_train[start:end]
        batch_y = Sig_train[start:end]
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    
    
    # Accuracy function:
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ######## ------- TESTING ----------############
    # Setup test data
    Var_test = df_test.iloc[:,1:nVar+1].values
    
    # Now the signal
    Sig_test_bit1 = df_test['signal'].values
    
    Sig_test_bit0 = Sig_test_bit1.copy()
    Sig_test_bit0 = 1 - Sig_test_bit0
    Sig_test = np.column_stack((Sig_test_bit0,Sig_test_bit1))
    print("\t Accuracy for alpha %.1E : %.3f" %(alpha,sess.run(accuracy, feed_dict={x: Var_test, y_: Sig_test})))
    
    # Get the weights
    weights = W.eval(session=sess)
    # Get probabilities assigned (i.e. evaluate y on test data)
    probs = y.eval(feed_dict = {x: Var_test}, session = sess)
    # now let's get the signal efficiency and background rejection on the test data
    print('\t Computing ROC curve ...')
    # build ROC curve by scanning over thresholds of probability of being
    # a background event and calculating signal efficiency/background rejection
    # at each threshold
    threshes = np.arange(0,1,0.01)
    Acceptance, Rejection = build_roc_curve(probs, Sig_test_bit1, threshes)

    return (probs,Acceptance,Rejection)


# ### Training and evaluating model

# In[91]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
alphas = np.logspace(-11,-12,1)
#fig = plt.figure()
#ax = fig.add_subplot(111)
it=0
for alpha in alphas:
    print("Training for alpha = %.2E"%alpha)
    c1 = 1.*( float(it) % 3.)/3.0
    c2 = 1.*( float(it) % 9.)/9.0
    c3 = 1.*( float(it) % 27.)/27.0
    probsSimple,accep,rej = runTensorFlowRegression(8,alpha)
    ax.scatter(accep,rej,c=[[c1,c2,c3]],label='Alpha: %.1E' %alpha)
    it+=1
  


# #### Although we did the running for a very small batch of training, we can see that Logistic regression will work good as a classification algorithm to distiguish signal versus background.

# # Artificial Neural Network

# In[70]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

neurons = [50, 100, 200, 300]
val_acc = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

for hidden in neurons:
  model = Sequential([
      Dense(hidden, input_shape=(18,), activation='relu'),
      Dense(hidden, activation='relu'),
      Dense(hidden, activation = 'relu'),
      Dense(1),
      Activation('sigmoid'),
  ])

  adam = optimizers.Adam(lr=1e-3)

  model.compile(optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=5)
  scores = model.evaluate(X_test, y_test)
  val_acc.append(scores[1])


# The processing time for my system to perform Artificial Neural NEtwork was so much that I refused to compile fully the program. 
# 
# Indeed, the process was heavier than the capability of my system, sorry!

# In[ ]:


hidden = 200
model = Sequential([
      Dense(hidden, input_shape=(18,), activation='sigmoid'),
      Dense(hidden, activation='sigmoid'),
      Dense(hidden, activation = 'sigmoid'),
      Dense(1),
      Activation('sigmoid'),
  ])

adam = optimizers.Adam(lr=1e-3)

model.compile(optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=1024)
scores = model.evaluate(X_test, y_test)
print("Validation accuracy: ", scores[1])


# Also , I liked to investigate the effect of Activation Functions on Neurons and the Neural Network, which is also time-consuming so I refuse to perform the code unfortunately.

# #### As a Result it seems to me that performing an artificial neural network on this huge data is not really reasonable.

# In[76]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

neurons = [2, 2]
val_acc = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

for hidden in neurons:
  model = Sequential([
      Dense(hidden, input_shape=(18,), activation='relu'),
      Dense(1),
      Activation('sigmoid'),
  ])

  adam = optimizers.Adam(lr=1e-3)

  model.compile(optimizer=adam,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  model.fit(X_train, y_train, validation_split=0.2, epochs=1, batch_size=1)
  scores = model.evaluate(X_test, y_test)
  val_acc.append(scores[1])


# ### Although we interrupted the kernel due to the processing being too much time-consuming.

# In[ ]:




