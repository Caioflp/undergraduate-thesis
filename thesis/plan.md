## My goal

Put together the cool things I have studied while participated in the
NPIV project in a single document, trying to link them together.
This includes some non-linear functional analysis, optimization, RKHS
methods, a bit of inverse problems theory and some econometrics
(instrumental variable regression).


## What I want to include

* Inverse problems
* Instrumental variables
* Kernel methods
    * RKHS theory, representer theorem, Mercer's theorem
    Maybe some theory of random elements on hilbert spaces


## Structure

* Introduction
    * Give a general perspective on the structure of the project,
      guiding the reader through the initial idea, which was to expand
      the SGD algorithm published in NIPS to NPIV regression, and how
      that naturally led to kernel methods.

* Inverse problems
    * Define inverse problems in general
    * Ill posedness of a problem
    * Integral equations of the first kind
    * Regularization methods
    * Landweber iteration for tikhonov regularization, comment on
      convergence

* Kernel Methods
    * Define RKHS and explain a bit of the theory
    * Explain the idea of Kernel Ridge regression
    * Vector RKHS, relationship with tensor products of Hilbert Spaces
      and Hilbert-Schmidt operators between these spaces.
    * Vector-valued kernel ridge regression

* Application in econometrics: Instrumental Variable Regression
    * Recap on what is instrumental variable regression
        - Explain confounding and show why OLS estimates are biased
        - Show a graphical example where endogeneity makes OLS be very
          wrong
        - Explain what is an instrumental variable and how it can be
          used to reduce bias.
        - Basic 2SLS
            - Explain the 2SLS algorithm and redo the confounded example with
              a 2SLS estimate.
    * IV and inverse problems
        - Link IV and inverse problems introducing the operator
          language we'll need later.
    * Newey paper
        - Explain the non-parametric idea of the paper, mentioning the
          things newey discusses which are useful to us, e.g., conditions
          which ensure identifiability.
    * KIV
        - Explain in some detail the Kernel IV model. Give more focus to
          the first stage procedure since this is what we are going to
          use later. State the convergence theorems.
        - Make connections to the kernel methods section, pointing out
          which results are being used.
    * Our method
        - Describe the whole method in detail and prove our convergence
          guarantees. Comment on the main differences between this and
          other methods.

* Appendixes if necessary


## Timeline

From 16/09 to 25/11. 10 weeks.

### Weeks 1 - 3

* Write the chapter about applications in IV.
Take note of interesting things about inverse problems and kernel methods
which would be nice to explain in the theoretical background chapters.

* (Done) Backtrack and decide what to talk about in each subject.
* (Done) Make a plan to write everything in ten weeks.
* (Done) Create a working template for the document.
* (Done) Start writing.
