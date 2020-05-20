# ICA_Report

# Blind Source Seperation using Independent Component Analysis (ICA)
In this report, we will construct an experiment focuses on the task of separating mixtures of sound signals into their underlying independent components using the technique of independent component analysis (ICA). Specifically, we will separate 3 audio mixes from 3 sources which are human voices and music by applying the fast-ICA algorithm. To denote these mathematically, consider x is mixed-audios, S is the source signals and A is mixing matrix. We have:

![x_formula](https://user-images.githubusercontent.com/63275375/82419666-e8d9ea00-9aa8-11ea-9236-c99af2a9bfc5.PNG)

Let W is inverse matrix of A. We will use fastICA algorithm to estimate W and use this formula to find S:

![S_formula](https://user-images.githubusercontent.com/63275375/82419664-e8d9ea00-9aa8-11ea-902c-b8c5bc41d157.PNG)


## Preprocessing data
Before apply fast ICA algorithm to estimate W, we first preprocessing our data. Pre-processing data in ICA is an important step, It will help you make the problem of ICA estimation simpler and better conditioned. There are two important pre-processing steps:
The first is **centering**. This is a simple subtraction of the mean from our input X. As a result the centered mixed signals will have zero mean which implies that also our source signals s are of zero mean. This step will simplify the ICA calculation.

```python
# Centering Data
def center(X):
    center=np.mean(X,axis=1,keepdims=True)
    X_center=X-center
    return X_center
X_center=center(X_)
print(X_center)
```
The second step is called **whitening**. The goal here is to linearly transform the observed signals X in a way that potential correlations between the signals are removed and their variances equal unity. As a result the covariance matrix of the whitened signals will be equal to the identity matrix. Whitening can be done by this formula:

![Whiten_formula](https://user-images.githubusercontent.com/63275375/82413905-677e5980-9aa0-11ea-8c4f-3a55fa2ab584.PNG)

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
Apply function to dataset and check the data after pre-preprocessing.
```python
X_center=center(X)
X_white=whiten(X_center)
print(np.round(np.cov(X_white)))
print(np.round(np.mean(X_white)))
```
## Implement the fast ICA algorithm
After data is pre-preprocessed, it is time to look at the fastICA algorithm. As discussed above one precondition for the ICA algorithm to work is that the source signals are non-Gaussian. Therefore, the result of the ICA should return sources that are as non-Gaussian as possible. In this experiment, we will choose negentropy as a measure of Gaussianity and the equation g, g' as follow :

![g_formula](https://user-images.githubusercontent.com/63275375/82419656-e7102680-9aa8-11ea-8bfa-948be188ab36.PNG)

```python
#Define g and g' function.
def g(X):
    g_x=np.tanh(X)
    return g_x
def g_d(X):
    g_d_x=1-(np.tanh(X))**2
    return g_d_x
```
The basic form of the fast-ICA algorithm as follows:

![FastICA_form](https://user-images.githubusercontent.com/63275375/82431091-68bb8080-9ab8-11ea-9a3b-bb6b676abeae.PNG)

In our experiment, we have more than 1 independent component so we have to decorrelate the output after each iterations. To achieve this, we use a deflation scheme based on a Gram–Schmidt-like decorrelation. We will estimate the independent component one by one and decorrelate output as follows:

![decorrlation](https://user-images.githubusercontent.com/63275375/82432144-d1572d00-9ab9-11ea-9fc6-ac4f1301f07e.PNG)

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
Now, all formulas are define, we will input the dataset and get the result.
```python
X_center=center(X)
X_white=whiten(X_center)
Estimate_S=Fastica(X_white,n_components=3,iterations=1000)
```
![Audio result](https://user-images.githubusercontent.com/63275375/82433062-1891ed80-9abb-11ea-8f32-523b337283f2.png)

The result looks very good, we get back all 3 sources.
