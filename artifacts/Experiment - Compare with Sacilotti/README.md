## Experiment 1 - Compare with Steck


### Methodology: 

* Remove movies with less than 10 interactions (we'll consider movies with move than
            10 reviews)
* Consider only users who have interacted with over 190 movies.
* Hardware limitation: we sampled only 300k rows from the resulting df from above
* Model trained: SVD++ using
  * 20 epochs
  * lr = 0.05
  * reg = 0.02
  * factors = 20