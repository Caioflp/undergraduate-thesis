## To do

* Write docstrings.

* Implement DGP of poster
* Use analytical version of p(z, x)/p(z)p(x)
* Use analytical version of E[Y|Z]
* Plot density estimates for sgd model.


* Plot things separetely.
* Improve plots so that things aren't as streched out.

* Implement the ML version.
* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$,
  or one of the other more recent IV estimation methods.
* Implement mirror descent

* Improve `Estimates` class so that I don't have to write every
  operation twice for grid and observed points.

* Perform hyperparameter sweeps.
* Decide if we'll keep mantaining the gd model.

### Meeting 10/05

* Implement DGP of poster
* Use analytical version of p(z, x)/p(z)p(x)
* Use analytical version of E[Y|Z]
* Implement mirror descent



## Done

* Implemented simple synthetic DGP.
* Implement our estimator with the projected loss
    - Ways to estimate $E [h(X)|Z = z]$: KNN, linear regression with
      basis expansion,
* Started using hydra from experiment management.
