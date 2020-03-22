# The Burgers Equation

## Introduction

We have obtained numerical solution to the Burgers equation using the Adams-Bashforth 2-step and 4-step methods. The time evolution of a sinusoidal initial condition has been plotted. The chosen time-step must be consistent with the CFL condition. With increasing time, the solution develops a shock discontinuity and the evolution beyond this point cannot be tracked with this code.

## Running the code

### AB2

The code for the AB 2-step scheme is stored in ```burgers_equation/AB2/burgersAB2.cxx```. The generated data-files are stored in ```burgers_equation/AB2/results``` and the image files and gnuplot scripts in ```burgers_equation/AB2/images```. To run the code, from within the directory ```burgers_equation/AB2/```, execute

```
g++ burgersAB2.cxx -o burgersAB2
./burgersAB2
```
This puts the data-files for each time step in ```results/```.

For plotting the results, go to the directory ```burgers_equation/AB2/images``` and run gnuplot with following command

```
gnuplot
load 'plot.gnu'
```
This generates a plot for the evolution of the initial condition in ```images/```.

### AB4

The code for the AB 4-step scheme is stored in ```burgers_equation/AB4/burgersAB4.cxx```. The generated data-files are stored in ```burgers_equation/AB4/results``` and the image files and gnuplot scripts in ```burgers_equation/AB4/images```. To run the code, from within the directory ```burgers_equation/AB4/```, execute

```
g++ burgersAB4.cxx -o burgersAB4
./burgersAB4
```
This puts the data-files for each time step in ```results/```.

For plotting the results, go to the directory ```burgers_equation/AB4/images``` and run gnuplot with following command

```
gnuplot
load 'plot.gnu'
```
This generates a plot for the evolution of the initial condition in ```images/```.

## Results

The following plots show the evolution of the initial condition _sin(2 pi x)_ with time. It can be seen that the solution gradually steepens resulting in a shock formation. Time varies from 0 to 0.15 in steps of 0.001 for both schemes. The plots show 5 equally spaced (in time)snapshots of the solution.

* AB 2-step

![image1](AB2/images/time_evolution2)

* AB 4-step

![image1](AB4/images/time_evolution)

