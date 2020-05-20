# ICA_Report

# Blind Source Seperation using Independent Component Analysis (ICA)
In this experiment, we will focuses on the task of separating mixtures of sound signals into their underlying independent components using the technique of independent component analysis (ICA). Cụ thể, chúng tối sẽ phân tách 3 hỗn hợp âm thanh được tạo từ 3 nguồn phát là tiếng nói người và tiếng nhạc hòa tấu bằng cách áp dụng thuật toán fastICA. To denote these mathematically, consider x is mixed signal, S is the matrix of source signals and A is mixing matrix. We have:

![x_formula](https://user-images.githubusercontent.com/63275375/82419666-e8d9ea00-9aa8-11ea-9236-c99af2a9bfc5.PNG)

Call W is inverse matrix of A. We will use fastICA algorithm to estimate W and use this formula to find S:

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
