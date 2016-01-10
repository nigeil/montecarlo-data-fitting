#This tests the searcher.py script in order to quantify how
#well it works in extreme and normal use cases.

import numpy as np
import pylab as pl
import inspect
import time 

from searcher import searcher, goodness 

#1 parameter functions
def line_1(x, a):
    return a * x

def sin_1(x, a):
    return a * np.sin(x)

def exp_1(x, a):
    return a * np.exp(x)

#2 parameter functions
def line_2(x, a,b):
    return a * x + b

def sin_2(x, a,b):
    return a * np.sin(b*x)

def exp_2(x, a,b):
    return a * np.exp(b*x)

#4 parameter functions
def poly_4(x, a,b,c,d):
    return a * x**3 + b * x**2 + c * x + d

def sin_4(x, a,b,c,d):
    return a * np.sin(b*x +c) + d

def exp_4(x, a,b,c,d):
    return a * np.exp(b * (x + c)) + d

#6 parameter functions
def poly_6(x, a,b,c,d,e,f):
    return (a * x**5 + b * x**4 + c* x**3 + d * x**2 
            + e * x + f)

def sin_6(x, a,b,c,d,e,f):
    return a * np.sin(b*x + c) + d * np.cos(e*x + f)

def exp_6(x, a,b,c,d,e,f):
    return a * np.exp(b*(x+c)) + d * np.exp(e*(x+f))

#8 parameter functions
def exp_sin_8(x, a,b,c,d,e,f,g,h):
    return a * np.exp(b*(x+c)) * (np.sin(d*x + e) + np.cos(f*x + g)) + h


#Get the tests together
test_functions = [line_1, sin_1, exp_1,
                  line_2, sin_2, exp_2,
                  poly_4, sin_4, exp_4,
                  poly_6, sin_6, exp_6,
                  exp_sin_8]

test_names = ["a * x", "a * sin(x)", "a * exp(x)",
              "a * x + b", "a * sin(b*x)", "a * exp(b*x)",
              "a * x**3 + b * x**2 + c * x + d", "a * sin(b*x + c) + d",
              "a * exp(b*(x+c)) + d", 
              "a * x**5 + b * x**4 + c* x**3 + d * x**2 + e * x + f",
              "a * sin(b*x + c) + d * cos(e*d + f)", 
              "a * exp(b*(x+c)) + d * exp(e*(x+f))",
              "a * np.exp(b*(x+c)) * (np.sin(d*x + e) + np.cos(f*x + g)) + h"]

#Generate parameters, as well as noisy data to fit, for each function
n_test_parameters = [len((inspect.getargspec(test_functions[i]))[0]) - 1 
                     for i in range(0, len(test_functions))]
n_tests_per_function = 4 #Get list of (n_tests_per_function) sets of parameters for each function
noise_level = 0.05 #Data will be in range [f(x)*(1-noise_level), f(x)*(1+noise_level)]

test_x = np.linspace(-1,1,100)
test_y = []
test_parameters = []
p_min = np.array([-2 for i in range(0,max(n_test_parameters))])
p_max = np.array([2 for i in range(0,max(n_test_parameters))])

for i in range(0,len(test_functions)):
    test_parameters.append([])
    test_y.append([])
    for j in range(0,n_tests_per_function):
        test_parameters[i].append(np.random.uniform(-1, 1, n_test_parameters[i]))
        test_y[i].append(test_functions[i](test_x, *(test_parameters[i][j])))
        test_y[i][j] *= (1 + np.random.uniform(-1 * noise_level, noise_level, len(test_x)))

#Run the tests and characterize the results
fits = []
guesses = []
degrees = []

for i in range(0, len(test_functions)):
    print(" ")
    print("Test: " + str(i) + " | f(x) = " + test_names[i])  
    tStart = time.time()
    fits.append([])
    guesses.append([])
    degrees.append([])
    for j in range(0,n_tests_per_function):
        fit, guess, degree  = searcher(test_functions[i], test_x, test_y[i][j], p_min, p_max)
        fits[i].append(fit)
        guesses[i].append(guess)
        degrees[i].append(degree)
    print("DEBUG: time taken =  " + str(time.time() - tStart) + " s")
    for j in range(0, n_tests_per_function):
        print("True parameters: " + str(test_parameters[i][j]))
        print("Best fit parameters: " + str(fits[i][j][0]))
        print("Guessed parameters: " + str(guesses[i][j]) + " | Degree: " + str(degrees[i][j]))
        pl.plot(test_x,test_y[i][j],'bo',alpha=0.2,label="Original data ")
        pl.plot(test_x,test_functions[i](test_x, *(fits[i][j][0])), 'g-', label="Best fit" )
        pl.plot(test_x,test_functions[i](test_x, *(guesses[i][j])),'m-', 
                label="Best guess," + " degree = " + str(degrees[i][j]))
        pl.legend(loc=2)
        pl.suptitle("f(x) = " + str(test_names[i]))
        pl.title("Goodness of guess = " 
                + str(goodness(test_functions[i],test_x,test_y,guesses[i][j])))
        #pl.title("True parameters = " + str(test_parameters[i][j]) + "\nGuessed parameters = " 
        #          + str(guesses[i][j]))
        pl.savefig("testPlots/" + test_names[i] + "_test" + str(j) + ".png")
        pl.clf()
