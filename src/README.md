## To do

* Implement the ML version.
* Try to estimate $ E[Y | Z] $ non parametrically, like Newey does
  (hermite polynomias and so forth).
* Keep translating the proofing and taking note of which assumptions must
  hold.

* Implement way to save parameters in each execution
* Setup a working logger

* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$,
  or one of the other more recent IV estimation methods.

* Implement Mirror Descent / Nesterov Acceleration

## What to show Yuri Rezende in 25/05 meeting

* Review algorithm formulation
    - Comment on exact gradient version and why it was discontinued
    - Explain how each term is being approximated
    - Comment on the possible gain from directly estimating the copula
* Show performance of algorithm
    - Multiple datasets (poster and Deep GMM)
    - Show influence of bandwidth parameter in density estimation
    - Show influence of K in KNN for estimate and Y projection
    - Show some other types of projectors (e.g. linear regression with
      basis expansion)
* Comment on main difficulties that arise when trying to adapt the proof
  to the IV case


## Meeting 16/05

* Main thing: Compile everything we've done so far to show Yuri.
* Run our algorithm (with estimates, KNN) in Deep GMM examples.
* Maybe choose K through validation.
* See if the available implementations of different models are easy to
  use.
* Try to fit our proof to the IV case.
* Implement the ML version.

* Maybe implement Mirror Descent or Nesterov Acceleration.


## Meeting 10/05

* Plot density estimates for sgd model.
* Implement DGP of poster
* Use analytical version of $ p(z, x)/p(z)p(x) $
* Use analytical version of $ E[Y|Z] $
* Implement mirror descent


## Done

* Use analytical version of p(z, x)/p(z)p(x) (Seems to perform worse?)
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
