## To do

* Verify the implementation

* Plot density estimates for sgd model.

* Use analytical version of p(z, x)/p(z)p(x)
* Use analytical version of E[Y|Z]

* Implement the ML version.
* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$,
  or one of the other more recent IV estimation methods.
* Implement mirror descent

* Perform hyperparameter sweeps.


### Meeting 10/05

* Implement DGP of poster
* Use analytical version of $ p(z, x)/p(z)p(x) $
* Use analytical version of $ E[Y|Z] $
* Implement mirror descent


## Done

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




FunctionalSGDEvaluator(model, dataset)
