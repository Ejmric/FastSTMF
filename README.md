# FastSTMF

Fast Sparse Tropical Matrix Factorization (FastSTMF)

FastSTMF is a novel approach for matrix completion based on sparse tropical matrix factorization (STMF). Please refer for the model's details to Omanović, A., Oblak, P. & Curk, T. FastSTMF: Efficient tropical matrix factorization algorithm for sparse data. Preprint at https://arxiv.org/ (2022).

### Real data
We used the real TCGA data in our experiments from the [paper by Rappoport N. and Shamir R.](https://academic.oup.com/nar/article/46/20/10546/5123392), and the data can be downloaded from the [link](http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html). Additional preprocessing before running our experiments is provided in our paper. PAM50 data can be found on the [link](https://github.com/CSB-IG/pa3bc/tree/master/bioclassifier\_R). BRCA subtypes are collected from [CBIO portal](https://www.cbioportal.org/).

### Use
```
import numpy.ma as ma
from FastSTMF import FastSTMF

data = ma.array(np.random.rand(100,100), mask=np.zeros((100,100)))
model = FastSTMF(rank=5, initialization="random_vcol", threshold=100)
model.fit(data)
approx = model.predict_all()
```

### Additional

The implementation of the "distance correlation" measure is from the following [link](https://gist.github.com/SherazKhan/4b2fe45c50a402dd73990c98450b2c89).
