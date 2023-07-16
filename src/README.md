## To do

* Finish proof:
    - Use bounds on convergence of density ration estimates.
    - Use bounds on Kernel Ridge Regression (for the expectation operator
      and for E[Y|Z]).
    - Formulate things in a "with high probability" way, not in an
      "average loss" way.
* Reimplement things using kernel methods.
* Benchmark against KIV, DeepIV, DeepGMM, 2SLS.


* Implement way to save parameters in each execution
* Setup a working logger
* Take a look at the IV datasets that Moises sent me.
* Implement the minimax estimator with a RKHS ball as $\mathcal{G}$,
  or one of the other more recent IV estimation methods.
* Implement Mirror Descent / Nesterov Acceleration


## Meeting 12/07

* Marcelo Moreira
* Next steps


## Meeting 07/06

* Redo all the calculations we did, in LaTeX (copy the neurips paper's
  template and work on top of that, pointing out what's different in the
  proof)
* Formulate and implement the algorithm with different datasets per
  iteration, and also different datasets for phi and $ (T, r_0) $ within
  each iteration.
* Think about what are the biases when estimating phi, T and r_0.

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

* Read material on density estimation and see if it gives us the results
  we need.
* Redo all the calculations we did, in LaTeX (copy the neurips paper's
  template and work on top of that, pointing out what's different in the
  proof)
* Keep translating the proof and taking note of which assumptions must
  hold.
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
