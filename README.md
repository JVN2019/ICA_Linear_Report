# ICA_MrSon

# Independent Component Analysis (ICA) implementation from scratch in Python

ICA is an efficient technique to decompose linear mixtures of signals into their underlying independent components. In this notebook, I will use fastICA algorithm to seperate mixing audio to original source audio.

Here we will start by first importing the necessary libraries and creating some toy signals which we will use to develop and test our ICA implementation.


```python
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Enable plots inside the Jupyter NotebookLet the
%matplotlib inline
```

### The generative model of ICA

The ICA is based on a generative model. This means that it assumes an underlying process that generates the observed data. The ICA model is simple, it assumes that some independent source signals *s* are linear combined by a mixing matrix A.

![png](images/01.png)

### Retrieving the components

The above equations implies that if we invert *A* and multiply it with the observed signals x we will retrieve our sources:

![png](images/02.png)


![png](images/03.png)

This means that what our ICA algorithm needs to estimate is *W*.


## Preprocessing functions

To get an optimal estimate of the independent components it is advisable to do some pre-processing of the data. In the following the two most important pre-processing techniques are explained in more detail.

The first pre-processing step we will discuss here is **centering**. This is a simple subtraction of the mean from our input X. As a result the centered mixed signals will have zero mean which implies that also our source signals s are of zero mean. This simplifies the ICA calculation and the mean can later be added back.


```python
def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered =  x - mean
    return centered, mean
```

For the second pre-processing technique we need to calculate the covariance. So lets quickly define it.


```python
def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n
```

The second pre-processing method is called **whitening**. The goal here is to linearly transform the observed signals X in a way that potential correlations between the signals are removed and their variances equal unity. As a result the covariance matrix of the whitened signals will be equal to the identity matrix


```python
def whiten(x):
    # Calculate the covariance matrix
    coVarM = covariance(X)

    # Single value decoposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, X)

    return Xw, whiteM
```

## Implement the fast ICA algorithm

Now it is time to look at the actual ICA algorithm. As discussed above one precondition for the ICA algorithm to work is that the source signals are non-Gaussian. Therefore the result of the ICA should return sources that are as non-Gaussian as possible. To achieve this we need a measure of Gaussianity. One way is Kurtosis and it could be used here but another way has proven more efficient. Nevertheless we will have a look at kurtosis at the end of this notebook.
For the actual algorithm however we will use the equations g and g'.

![png](images/07.png)

These equations allow an approximation of negentropy and will be used in the below ICA algorithm which is [based on a fixed-point iteration scheme](https://homepage.math.uiowa.edu/~whan/072.d/S3-4.pdf):

![png](images/08.png)


So according to the above what we have to do is to take a random guess for the weights of each component. The dot product of the random weights and the mixed signals is passed into the two functions g and g'. We then subtract the result of g' from g and calculate the mean. The result is our new weights vector. Next we could directly divide the new weights vector by its norm and repeat the above until the weights do not change anymore. There would be nothing wrong with that. However the problem we are facing here is that in the iteration for the second component we might identify the same component as in the first iteration. To solve this problem we have to decorrelate the new weights from the previously identified weights. This is what is happening in the step between updating the weights and dividing by their norm.


```python
def fastIca(signals,  alpha = 1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
            w = W[c, :].copy().reshape(m, 1)
            w = w / np.sqrt((w ** 2).sum())

            i = 0
            lim = 100
            while ((lim > thresh) & (i < iterations)):

                # Dot product of weight and signal
                ws = np.dot(w.T, signals)

                # Pass w*s into contrast function g
                wg = np.tanh(ws * alpha).T

                # Pass w*s into g prime
                wg_ = (1 - np.square(np.tanh(ws))) * alpha

                # Update weights
                wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

                # Decorrelate weights              
                wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.sqrt((wNew ** 2).sum())

                # Calculate limit condition
                lim = np.abs(np.abs((wNew * w).sum()) - 1)

                # Update weights
                w = wNew

                # Update counter
                i += 1

            W[c, :] = w.T
    return W
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

## Kurtosis

So finally lets check one last thing: The kurtosis of the signals.
Kurtosis is the fourth moment of the data and measures the "tailedness" of a distribution. A normal distribution has a value of 3, a uniform distribution like the one we used in our toy data has a kurtosis < 3. The implementation in Python is straight forward as can be seen from the code below which also calculates the other moments of the data. The first moment is the mean, the second is the variance, the third is the skewness and the fourth is the kurtosis. Here 3 is subtracted from the fourth moment so that a normal distribution has a kurtosis of 0.


```python
# Calculate Kurtosis

def kurt(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3

    return kurt, skew, var, mean
```

As we can see in the following all of our mixed signals have a kurtosis of ≤ 1 whereas all recovered independent components have a kurtosis of 1.5 which means they are less Gaussian than their sources. This has to be the case since the ICA tries to maximize non-Gaussianity. Also it nicely illustrates the fact mentioned above that the mixture of non-Gaussian signals will be more Gaussian than the sources.


```python
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(10,10))
for i in range(X.shape[0]):

    sns.kdeplot(X[i, :], lw=5, label='Mixed Kurtosis={}'.format(np.round(kurt(X[i, :])[0], decimals=1)))

for i in range(X.shape[0]):   
    sns.kdeplot(unMixed[i, :], lw=5, ls='--', label='unMixed Kurtosis={}'.format(np.around(kurt(unMixed[i, :])[0], decimals=1)))

ax.tick_params(labelsize=12)
ax.set_xlabel('value', fontsize=20)
ax.set_ylabel('count', fontsize=20)
ax.set_title('KDE plot of ', fontsize=25)
plt.show()
```


![png](images/output_29_0.png)



```python

```

