# The Heat Equation

## Introduction

We have used the Crank-Nicolson scheme to solve for the evolution of a given initial condition over a 1-D domain. The program file is called ```heateq.cxx```, the resulting data-files are stored in ```heat_equation/results/``` and the image files in ```heat_equation/images/```. Each data file contains the data for one time step. A comparison of the exact solution at a given time step with the numerical output has been plotted using _gnuplot_. 

## Execution


* Run the code


From within ```heat_equation/```, execute

```
g++ heateq.cxx -o heateq
```

This creates the object file ```heateq```. Run the file using 
```
./heateq
```


* Plot the results


The plotting scripts for _gnuplot_ are stored in ```images/```. From within ```images/```, execute
```
gnuplot
load 'plot.gnu'
load 'plotSigma.gnu'
```
This will plot two plots containing a comparison of numerical and exact results for various times. One plot shows the evolution of the intial condition over the domain. This allows us to visualize the diffusion process. The other plot is a semi-log plot of the solution at a fixed gridpoint allowing us to predict the diffusion constant from the value of the slope of the line.

## Results

A comparison of the exact and numerical solutions at different timesteps

![image](images/time_evolution)

Semi-Log plot of _u(x0) vs t_ at a fixed grid-point (= N-1 / 8) gives a straight line.

![image](images/sigma)

The value of the diffusion constant (sigma) can be estimated from the slope of the lines in the above figure. The red line, corresponding to the exact solution has a slope of -15.7913 while the numerical (blue) line has a slope of -15.8033. Therefore, the computed value of the diffusion constant, *sigma = 15.8033*.

The error is 1.2%.
