#Monte Carlo fitting function parameter searcher

##What does it do?

Fitting a model to a set of data can be challenging if one does not already have
a good guess at the optimal parameters for the function. This issue is exacerbated
by having a large number of meaningful (and thus non-removable) parameters in 
the model. 

The purpose of this routine is to:

1. Select parameter sets at random and calculate the RMSD of the fit line with
the data
2. Perform network analysis on the graph of these points to determine the best
possible guesses
3. Use these best guesses to generate the correct parameter set using
standard curve-fitting routines (scipy.optimize.curve\_fit)

##Who is it for?

If you have data and want to fit a model to it, but don't feel like intuiting 
a good guess at the values of parameters in your model, this is for you. 
Shotgun method - slow, inefficient, but takes next to no effort.

##How can I use it?

This software is still in an alpha stage and is being tested and refined
fairly often by none other than myself. If you'd like to play around with
it, you can run the test.py script to see how it manages to fit functions
with 1 to 10 free parameters.

More information to come.



