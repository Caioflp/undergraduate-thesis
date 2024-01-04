## Paper outline


## To do

* Optimize SAGDIV implementation
    * Keep using KIV first stage?
* Revise KIV implementation

* Benchmark against:
    * KIV - Revise
    * DeepGMM
    * DeepIV?
    * SmoothIV?
    * 2SLS?
    * Include performance of non IV method?

* Implement version for binary outcomes
    * Compare with ??? See what methods can address binary outcomes

* Setup a working logger


## Meeting 19/07/23

* Difficulty understanding KIV stage 1
* Strange assumptions
* Not sure if using KIV's first stage is a good thing
* Don't know how to bound \hat{\Phi} RKHS norm
* Don't know how to regress Y on Z


## Meeting 12/07/23

* Marcelo Moreira
* Next steps


## Meeting 07/06/23

* Redo all the calculations we did, in LaTeX (copy the neurips paper's
  template and work on top of that, pointing out what's different in the
  proof)
* Formulate and implement the algorithm with different datasets per
  iteration, and also different datasets for phi and $ (T, r_0) $ within
  each iteration.
* Think about what are the biases when estimating phi, T and r_0.

## What to show Yuri Rezende in 25/05/23 meeting

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


## Meeting 16/05/23

* Main thing: Compile everything we've done so far to show Yuri.
* Run our algorithm (with estimates, KNN) in Deep GMM examples.
* Maybe choose K through validation.
* See if the available implementations of different models are easy to
  use.
* Try to fit our proof to the IV case.
* Implement the ML version.

* Maybe implement Mirror Descent or Nesterov Acceleration.


## Meeting 10/05/23

* Plot density estimates for sgd model.
* Implement DGP of poster
* Use analytical version of $ p(z, x)/p(z)p(x) $
* Use analytical version of $ E[Y|Z] $
* Implement mirror descent


## Done

* Rewrite theory section of paper according to observations made in the
  document.
* Things that can be better:
    - Choosing kernel through validation
    - Choosing lengthscale through validation
* Implementation changes:
    - Use three datasets: (X, Z), (Y, Z), (Z), although the first two
      don't need to be independent from each other.
    - Crop h inside [-M, M] after each update
    - Write tests for uLSIF implementation
    - Implement \hat{ r_0 } and \hat{ T } using kernel methods
* Implement \hat{ Phi } using kernel methods
    - Read and underestand the closed form solutions
    - Understand how the regularization parameter is being chosen
* Understand how to apply KIV's stage 1
* Proof adaptation
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
