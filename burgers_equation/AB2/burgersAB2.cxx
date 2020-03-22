#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>

#define PI 3.14159
#define N 101

//prints a vector
void print_vector(double X[N]){
    for(int i = 0; i < N; i++)
        std::cout<<X[i]<<std::endl;
}

//create an N point grid
void grid(double X[N]){
    double dx = 1.0/(N-1) ;
    X[0] = 0;
    for (int i = 1; i < N; i++){
        X[i] = X[i-1] + dx;
    }
}

//sets the intial condition as sin(2 pi x)
void set_initial_condition(double U0[N], double X[N]){
    for (int i = 0; i < N; i++){
        U0[i] = sin(2* PI* X[i]);
    }
}

//The Adams-Bashforth 2 step scheme with periodic boundary-conditions
void AB2(const double U0[N], const double U1[N], double U2[N], double dt, double dx){

    U2[0] = U1[0] + (dt/(4*dx))*(U0[0]*(U0[1] - U0[N-2]) - 3*U1[0]*(U1[1] - U1[N-2]));
    U2[N-1] = U1[N-1] + (dt/(4*dx))*(U0[N-1]*(U0[1] - U0[N-2]) - 3*U1[N-1]*(U1[1] - U1[N-2]));

    for(int j = 1; j < N-1; j++){
        U2[j] = U1[j] + (dt/(4*dx))*(U0[j]*(U0[j+1] - U0[j-1]) - 3*U1[j]*(U1[j+1] - U1[j-1]));
    }
}

//copy array A to array B
void update(const double A[N], double B[N]){
    for(int i = 0; i < N; i++)
    B[i] = A[i];
}

//write arrays X and U to file solution_n.txt
void write(double X[N], double U[N], int a)
{
    char filename[150];
    sprintf(filename, "results/solution_%d.txt", a);
    std::ofstream file1;
    file1.open(filename);
    for(int i = 0; i < N; i++)
        file1<<X[i]<<"\t"<<U[i]<<std::endl;

    file1.close();
}

int main (){

    double X[N];  //the grid
    double U0[N]; //initial condition
    double U1[N]; //solution at (n+1)th time step
    double U2[N]; //solution at (n+2)th time step

    double dx = 1.0/(N-1); //spatial discretization
    double T = 0.15;       //total time for evolution
    double dt = 0.001;     //time step
    double t = 0.0;        //initial time
    int i=0;
    
    grid(X);

    //set initial condition and write it to file
    set_initial_condition(U0, X);
    write(X, U0, 0);

    //take the first time step using Euler's method
    U1[0] = U0[0] - (dt/(2*dx)) * U0[0] * (U0[1] - U0[N-2]);
    U1[N-1] = U0[N-1] - (dt/(2*dx)) * U0[N-1] * (U0[1] - U0[N-2]);

    for(int j = 1; j < N-1; j++){
    U1[j] = U0[j] - (dt/(2*dx)) * U0[j] * (U0[j+1] - U0[j-1]);
    }
    //write the first step (U1) to file
    write(X, U1, 1);

    //Adams-Bashforth 2 step method
    while(t < T){
        i++;
        //use U0 and U1 to get U2
        AB2(U0, U1, U2, dt, dx);
        //write U2 to file
        write(X, U2, i);
        //copy u1 -> u0
        update(U1, U0);
         //copy u2 -> u1
        update(U2, U1);
        t = t+dt;
    }

    return 0;

}