#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>

#define PI 3.14159
#define N 101 //grid size

void print_vector(double X[N]){
    for(int i = 0; i < N; i++)
        std::cout<<X[i]<<std::endl;
}

//create the grid
void grid(double X[N]){
    double dx = 1.0/(N-1) ;
    X[0] = 0;
    for (int i = 1; i < N; i++){
        X[i] = X[i-1] + dx;
    }
}

//set the initial condition as sin(2 PI x)
void set_initial_condition(double U0[N], double X[N]){
    for (int i = 0; i < N; i++){
        U0[i] = sin(2* PI* X[i]);
    }
}

//Adams-Bashforth 4-step scheme with periodic boundary condtions
void AB4(const double U0[N], const double U1[N],const double U2[N],
         const double U3[N], double U4[N], double dt, double dx){

    //Periodic boundary conditions
    U4[0] = U3[0] + (dt/(48*dx))*(-55* U3[0]*(U3[1] - U3[N-2])
                                      +59* U2[0]*(U2[1] - U2[N-2])
                                      -37* U1[0]*(U1[1] - U1[N-2])
                                      +9* U0[0]*(U0[1] - U0[N-2]));

    U4[N-1] = U4[0];

    //Adams-Bashforth 4-step scheme
    for(int j = 1; j < N-1; j++){
        U4[j] = U3[j] + (dt/(48*dx))*(-55* U3[j]*(U3[j+1] - U3[j-1])
                                      +59* U2[j]*(U2[j+1] - U2[j-1])
                                      -37* U1[j]*(U1[j+1] - U1[j-1])
                                      +9* U0[j]*(U0[j+1] - U0[j-1]));
    }
}

//copy A to B
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

    double X[N];
    double U0[N];
    double U1[N];
    double U2[N];
    double U3[N];
    double U4[N];

    double dx = 1.0/(N-1);
    double T = 0.15;
    double dt = 0.001;
    double t = 0.0;
    int i = 0;

    grid(X);

    set_initial_condition(U0, X);
    write(X, U0, 0);

    //get U1, U2 and U3 using Euler's method    
    U1[0] = U0[0] - (dt/(2*dx)) * U0[0] * (U0[1] - U0[N-2]);
    U1[N-1] = U0[N-1] - (dt/(2*dx)) * U0[N-1] * (U0[1] - U0[N-2]);

    for(int j = 1; j < N-1; j++){
    U1[j] = U0[j] - (dt/(2*dx)) * U0[j] * (U0[j+1] - U0[j-1]);
    }
    write(X, U1, 1);

    U2[0] = U1[0] - (dt/(2*dx)) * U1[0] * (U1[1] - U1[N-2]);
    U2[N-1] = U1[N-1] - (dt/(2*dx)) * U1[N-1] * (U1[1] - U1[N-2]);

    for(int j = 1; j < N-1; j++){
    U2[j] = U1[j] - (dt/(2*dx)) * U1[j] * (U1[j+1] - U1[j-1]);
    }
    write(X, U2, 2);

    U3[0] = U2[0] - (dt/(2*dx)) * U2[0] * (U2[1] - U2[N-2]);
    U3[N-1] = U2[N-1] - (dt/(2*dx)) * U2[N-1] * (U2[1] - U2[N-2]);

    for(int j = 1; j < N-1; j++){
    U3[j] = U2[j] - (dt/(2*dx)) * U2[j] * (U2[j+1] - U2[j-1]);
    }
    write(X, U3, 3);

    //Adams-Bashforth 4-step scheme for time marching
    while(t < T){
        i++;
        AB4(U0, U1, U2, U3, U4, dt, dx);
        write(X, U4, i);
        update(U1, U0); //u1 -> u0
        update(U2, U1); //u2 -> u1
        update(U3, U2); //u3 -> u2
        update(U4, U3); //u4 -> u3
        t = t+dt;
        std::cout<<"t = "<<t<<std::endl;
    }

    return 0;

}