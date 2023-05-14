## To do

* Use analytical version of p(z, x)/p(z)p(x)
* Use analytical version of E[Y|Z]

* Implement way to save parameters in each execution
* Setup a working logger

* Implement the ML version.
* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$,
  or one of the other more recent IV estimation methods.
* Implement mirror descent

* Perform hyperparameter sweeps.


### Meeting 10/05

* Plot density estimates for sgd model.
* Implement DGP of poster
* Use analytical version of $ p(z, x)/p(z)p(x) $
* Use analytical version of $ E[Y|Z] $
* Implement mirror descent


## Done

* Verify the implementation (seems to be correct)
* Remove separate plots for grid and observed points in functional sgd.
* Move ploting to separate module.
* Decide if we'll keep mantaining the gd model.
* Improve `Estimates` class so that I don't have to write every
  operation twice for grid and observed points.
* Plot things separetely.
* Improve plots so that things aren't as streched out.
* Implement poster's DGP
* Write docstrings.
* Started using hydra from experiment management.
* Implement our estimator with the projected loss
    - Ways to estimate $E [h(X)|Z = z]$: KNN, linear regression with
      basis expansion,
* Implemented simple synthetic DGP.
