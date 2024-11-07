## Experiment 1 - Compare with Steck


### Methodology: 

* Only consider movies with rating >= 4
* Remove movies without gender
* Hardware limitation: we sampled only 300k rows from the resulting df from above
* Model trained: SVD++ using
  * 20 epochs
  * lr = 0.05
  * reg = 0.02
  * factors = 20


### Possible improvements
* The author uses NMF. Might be worth running it again with NMF
