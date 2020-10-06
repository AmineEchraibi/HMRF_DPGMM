# Hidden Markov Random Field Dirichlet Process Gaussian Mixture Model for Semi-Supervised Clustering

In the following we explain some details of Hidden Markov Random Field Dirichlet Process Gaussian Mixture Model, for more details please visit (https://hal.archives-ouvertes.fr/hal-02372337/document)
## Graphical model

![alt text](images/hmrfdpgmm.png?raw=true)

## Mathematical model

![alt text](images/math1.png?raw=true)
![alt text](images/math2.png?raw=true)
![alt text](images/math3.png?raw=true)

## Variational Inference

![alt text](images/var2.png?raw=true)

## Code instructions

```python
from model import HMRF_GMM
hmrf_dpgmm = HMRF_GMM(Data, Truncation_level, Partial_labels)
hmrf_dpgmm.Inference()
Clusters = hmrf_dgpgmm.infer_clusters()
```
