## To do

* Improve `Estimates` class so that I don't have to write every operation
  twice for grid and observed points.
* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$.
* Write docstrings.
* Improve plots so that things aren't as streched out.
* Plot density estimates for sgd model.
* Decide if we'll keep mantaining the gd model.
* Implement the ML version.
* Perform hyperparameter sweeps.


## Done

* Implemented simple synthetic DGP.
* Implement our estimator with the projected loss
    - Ways to estimate $E [h(X)|Z = z]$: KNN, linear regression with
      basis expansion,
* Started using hydra from experiment management.
