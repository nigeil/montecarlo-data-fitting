#!/bin/python
#Searches for a 'region' of good parameters that are densely connected
#then minimizes the fitting function error using standard methods. 
#Useful for fitting functions with 4+ parameters where intial 
#guesses are difficult to come by.

import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
import inspect

### AGNOSTIC FUNCTIONS ###

#How good the fit is. [0,1] where 1 is the best fit.
#Calculate RMSD of points with predicted values from model
def goodness(f,x,y,p):
    sum_of_squares = np.sum(np.square(y - f(x,*p)))
    ret = np.sqrt(sum_of_squares/len(x))
    return ret

### MAIN FUNCTIONS ###
def searcher(func, x, y, p_min, p_max, max_itr=None, n_interp=4):
    n_params = len(inspect.getargspec(func)[0]) - 1 
    if max_itr is None:
        if n_params <= 3:
            max_itr = 10**(5)
        else:
            max_itr = 10**(6)
    ###Searching for initial points to serve as nodes of graph

    scores = np.zeros(max_itr)
    tested_parameters = np.array([np.empty(n_params) for i in range(0,max_itr)]) 

    for i in range(0,max_itr):
        tested_parameters[i] = np.array([np.random.uniform(p_min[j], p_max[j]) 
                                        for j in range(0,n_params)])
        scores[i] = goodness(func,x,y,tested_parameters[i])

    thresh_score = np.min(scores)
    high_thresh_mult = 2 
    low_thresh_mult = 0

    nodes = []  #good parameter sets, to serve as nodes on our graph
    nodes_scores = []
    for i in range(0, len(tested_parameters)):
        if (scores[i] < thresh_score * high_thresh_mult 
            and scores[i] > thresh_score * low_thresh_mult):
            nodes.append(tested_parameters[i])
            nodes_scores.append(scores[i])

    ###Begin testing for connectedness using interpolating points between nodes
    edges = []
    for i in range(0,len(nodes)):
        for j in range(i + 1,len(nodes)):
            interpolating_pts = []
            add_edge = True
            for k in range(0,n_interp):
                interpolating_pts.append(nodes[i] + (k+1) * (nodes[j] - nodes[i]) / (n_interp+1))
            for k in range(0,n_interp):
                score = goodness(func, x, y, interpolating_pts[k])
                if not (score < thresh_score * high_thresh_mult 
                        and score > thresh_score * low_thresh_mult):
                    add_edge = False
                    break
            if add_edge is True:
                edges.append([i,j])

    ###Calculate degree of each node
    #Take node with highest degree as best testing point
    degrees = np.zeros(len(nodes))
    if len(edges) > 1:
        for edge in edges:
            degrees[edges[0]] += 1
            degrees[edges[1]] += 1
    else: #failed to find anything useful
        bestGuess = nodes[-1]
        bestDegree = -1 
        p_fit = [bestGuess,[]] 
        return p_fit, bestGuess, bestDegree

    bestGuess = np.empty(n_params)
    bestDegree = 0
    bestNodeIndex = 0
    for i in range(0,len(nodes)):
        if degrees[i] > bestDegree:
            bestNodeIndex = i
            bestDegree = degrees[i]
            bestGuess = nodes[bestNodeIndex]


    ###Use best guess to try and get good fitting parameters
    try:
        p_fit = curve_fit(func,x,y,p0=bestGuess)
    except RuntimeError:
        p_fit = [bestGuess * -1, []]
    return p_fit, bestGuess, bestDegree


