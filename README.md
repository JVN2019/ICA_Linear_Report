# ICA_MrSon

# Independent Component Analysis (ICA) implementation from scratch in Python

ICA is an efficient technique to decompose linear mixtures of signals into their underlying independent components. In this notebook, I will use fastICA algorithm to seperate mixing audio to original source audio.

Here we will start by first importing the necessary libraries and creating some toy signals which we will use to develop and test our ICA implementation.


```python
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
np.random.seed(42)
%matplotlib inline
```

### The generative model of ICA

The ICA is based on a generative model. This means that it assumes an underlying process that generates the observed data. The ICA model is simple, it assumes that some independent source signals *s* are linear combined by a mixing matrix A.

### Import Dataset

```python
SR1, source1 = wavfile.read('beet.wav')
SR2, source2 = wavfile.read('beet9.wav')
SR3, source3 = wavfile.read('mike.wav')
MSR1, mix_source1 = wavfile.read('mixing1.wav')
MSR2, mix_source2 = wavfile.read('mixing2.wav')
MSR3, mix_source3 = wavfile.read('mixing3.wav')
n_samples=mix_source1.shape[0]
noise1 = 1.5*np.random.normal(loc=0.0,scale=1000, size=n_samples)
noise2 = 0.7*np.random.normal(loc=0.0, size=n_samples)
noise3 = 2.1*np.random.normal(loc=0.0, size=n_samples)
Source=np.asarray([source1,source2,source3])
X_=np.asarray([mix_source1+noise1,mix_source2,mix_source3])

fig, ax = plt.subplots(2, 1, figsize=[18, 5], sharex=True)
for i in range(len(Source)):
    ax[0].plot(Source[i])
ax[1].plot(X_[0])
ax[0].set_title('Original Source', fontsize=25)
ax[0].tick_params(labelsize=12)
ax[1].set_xlabel('Sample number', fontsize=20)
plt.show()
```

## Preprocessing functions

To get an optimal estimate of the independent components it is advisable to do some pre-processing of the data. In the following the two most important pre-processing techniques are explained in more detail.

The first pre-processing step we will discuss here is **centering**. This is a simple subtraction of the mean from our input X. As a result the centered mixed signals will have zero mean which implies that also our source signals s are of zero mean. This simplifies the ICA calculation and the mean can later be added back.


```python
# Centering Data
def center(X):
    center=np.mean(X,axis=1,keepdims=True)
    X_center=X-center
    return X_center
X_center=center(X_)
print(X_center)
```
The second pre-processing method is called **whitening**. The goal here is to linearly transform the observed signals X in a way that potential correlations between the signals are removed and their variances equal unity. As a result the covariance matrix of the whitened signals will be equal to the identity matrix


```python
def white(X):
#Calculate covariance matrix of X
    covariance=np.cov(X)
#Calculate eigenvalue and eigenvector
    d, E = np.linalg.eigh(covariance)
    D = np.diag(d)
#Whiten data
    x_white=np.dot(E,np.dot(np.sqrt(np.linalg.inv(D)),np.dot(E.T,X)))
    return x_white
X_white=white(X_center)
print(X_white)
```

## Implement the fast ICA algorithm

Now it is time to look at the actual ICA algorithm. As discussed above one precondition for the ICA algorithm to work is that the source signals are non-Gaussian. Therefore the result of the ICA should return sources that are as non-Gaussian as possible. To achieve this we need a measure of Gaussianity. One way is Kurtosis and it could be used here but another way has proven more efficient. Nevertheless we will have a look at kurtosis at the end of this notebook.
For the actual algorithm however we will use the equations g and g'.
\begin{equation}g(w^TX)=\tanh \left(W^T X\right)\space\space g'(W^TX)=1-\tanh^2 \left(W^T X\right)\end{equation}
```python
#Define g and g' function.
def g(X):
    g_x=np.tanh(X)
    return g_x
def g_d(X):
    g_d_x=1-(np.tanh(X))**2
    return g_d_x
```
These equations allow an approximation of negentropy and will be used in the below ICA algorithm which is [based on a fixed-point iteration scheme]:




So according to the above what we have to do is to take a random guess for the weights of each component. The dot product of the random weights and the mixed signals is passed into the two functions g and g'. We then subtract the result of g' from g and calculate the mean. The result is our new weights vector. Next we could directly divide the new weights vector by its norm and repeat the above until the weights do not change anymore. There would be nothing wrong with that. However the problem we are facing here is that in the iteration for the second component we might identify the same component as in the first iteration. To solve this problem we have to decorrelate the new weights from the previously identified weights. This is what is happening in the step between updating the weights and dividing by their norm.


```python
def Fastica(X, iterations,n_components=-1, tolerance=1e-5):
    if n_components < 1 :
        n_components = X.shape[0]
#Create random intial value for w
    W = np.zeros((n_components, n_components), dtype=X.dtype)

    for i in range(n_components):
        
        w = np.random.rand(n_components)
#Calculate and update new value for w        
        for j in range(iterations):
            
            w_new = new_w(w, X)
#Decorrelate output when use fastICA for many units.            
            if i > 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
#Check conrvege            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            
            w = w_new
            
            if distance < tolerance:
                break
                
        W[i, :] = w
#Calculate ICs       
    S = np.dot(W, X)
    
    return S
```

### Pre-processing

So... before we run the ICA we need to do the pre-processing.


```python
# Center signals
Xc, meanX = center(X)

# Whiten mixed signals
Xw, whiteM = whiten(Xc)
```

### Check whitening

Above we mentioned that the covariance matrix of the whitened signal should equal the identity matrix:

![png](images/05.png)

...and as we can see below this is correct.


```python
# Check if covariance of whitened matrix equals identity matrix
print(np.round(covariance(Xw)))
```

    [[ 1. -0.  0.]
     [-0.  1.  0.]
     [ 0.  0.  1.]]


### Running the ICA

Finally... it is time to feed the whitened signals into the ICA algorithm.


```python
W = fastIca(Xw,  alpha=1)

#Un-mix signals using
unMixed = Xw.T.dot(W.T)

# Subtract mean
unMixed = (unMixed.T - meanX).T
```


```python
# Plot input signals (not mixed)
fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(S, lw=5)
ax.tick_params(labelsize=12)
ax.set_xticks([])
ax.set_yticks([-1, 1])
ax.set_title('Source signals', fontsize=25)
ax.set_xlim(0, 100)

fig, ax = plt.subplots(1, 1, figsize=[18, 5])
ax.plot(unMixed, '--', label='Recovered signals', lw=5)
ax.set_xlabel('Sample number', fontsize=20)
ax.set_title('Recovered signals', fontsize=25)
ax.set_xlim(0, 100)

plt.show()
```


![png](images/output_24_0.png)



![png](images/output_24_1.png)


The result of the ICA are plotted above, and the result looks very good. We got all three sources back!
