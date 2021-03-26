#include <iostream>
#include <math.h>
#include <fstream>

#define N 101
#define PI 3.14159

//create an array of the grid points
void create_grid(double X[N], double dx){
    X[0] = 0.0;
    for(int i = 0; i < N; i++)
        X[i + 1] = X[i] + dx;
}

//function to define the initial condition
void set_initial_condition(double u0[N]){
//u0[i] = sin(4 pi x)
    for(int i = 0; i < N; i++)
        u0[i] = sin(4*PI*i / (N-1));
}

//set the values at the boundary
void set_boundary_conditions(double u0[N])
{
//Dirichilet boundary conditions
    u0[0] = 0;
    u0[N] = 0;
}

//obtain the RHS of the system of equations due to implicit scheme
void create_RHS(double b[N], double u0[N], double alpha)
{
    u0[0] = 0.0;
    u0[N-1] = 0.0;
    for(int i = 1; i < N-1; i++)
        b[i] = alpha * (u0[i - 1] + u0[i + 1]) + (1 - 2 * alpha) * u0[i];
}

//construct the tri-diagonal matrix
void create_Tri_Diagonal_Matrix(double A[N][N], double alpha)
{
    for(int i = 1; i < N-1; i++)
        {
            A[i][i - 1] = -alpha;
            A[i][i] = 1 + 2 * alpha;
            A[i][i + 1] = -alpha;
        }

    A[0][0] = 1.0;
    A[N-1][N-1] = 1.0;

}

//get diagonals in a form usable by the TDMA solver

//    |b0 c0 0 ||x0| |d0|
//    |a1 b1 c1||x1|=|d1|
//    |0  a2 b2||x2| |d2|

//lower diagonal (D1) = 0, a1, a2
//middle diagonal (D2) = b1, b2, b3
//upper diagonal (D3) = c1, c2, 0

void put_diagonals_in_arrays(double A[N][N], double D1[N], double D2[N], double D3[N])
{
    for(int i = 0; i < N; i++)
    {
        D1[i] = 0.0;
        D2[i] = 0.0;
        D3[i] = 0.0;
    }

    for(int i = 0; i < N; i++) //assign elements in the diagonal arrays
	D1[i+1] = A[i+1][i];

	for(int i = 0; i <= N; i++) //assign elements in the diagonal arrays
	D2[i] = A[i][i];

	for(int i = 0; i < N-1; i++) //assign elements in the diagonal arrays
	D3[i] = A[i][i + 1];

}

//The TDMA Solver - a,b,c are the diagonals
//output is stored in d
void solve(double* a, double* b, double* c, double* d, int n) {
     n--; // since we start from x0 (not x1)
    c[0] /= b[0];
    d[0] /= b[0];

    for (int i = 1; i < n; i++) {
        c[i] /= b[i] - a[i]*c[i-1];
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
    }

    d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

    for (int i = n; i-- > 0;) {
        d[i] -= c[i]*d[i+1];
    }
}

//copy array b to array u
void update(double b[N], double u[N]){
    for(int i = 0; i < N; i++)
        u[i] = b[i];
}

//get the exact solution (at time t) and write it to a file
void exact_solution(double exact[N], double X[N], double t, double kappa){
    for(int i = 0; i < N; i++)
        exact[i] = exp(-kappa *16*PI*PI * t)*sin(4*PI*X[i]);

    char filename[150];
    sprintf(filename, "results/exact_solution.txt");
    std::ofstream file1;
    file1.open(filename);
    for(int i = 0; i < N; i++)
    file1<<i<<"\t"<<X[i]<<"\t"<<exact[i]<<std::endl;

    file1.close();
}

//write solution at the current time step to a file
void write(double X[N], double u0[N], double exact[N], int a)
{
    char filename[150];
    sprintf(filename, "results/solution_%d.txt", a);
    std::ofstream file1;
    file1.open(filename);
    for(int i = 0; i < N; i++)
        file1<<i<<"\t"<<X[i]<<"\t"<<u0[i]<<"\t"<<exact[i]<<std::endl;

    file1.close();
}

//write function to get sigma
//write values at the nth node at every time step to a file
void write_at_fixed_gridpoint(double u[N], double u0[N], double exact, double t, int n)
{
    char namefile[150];
    sprintf(namefile, "results/sigma.txt");
    std::ofstream file1;
    file1.open(namefile, std::ios_base::app);
    file1 << t <<"\t"<< u[n] / u0[n] <<"\t"<< exact / u0[n] <<std::endl;

    file1.close();
}

//function to print a vector
void display_vector(double a[N])
{
    for(int i = 0; i < N; i++)
        std::cout<<a[i]<<std::endl;
}

//function to print a matrix
void display_matrix(double a[N][N])
{
    for(int j = 0; j < N; j++){
        for(int i = 0; i < N; i++)
            std::cout<<a[j][i]<<"\t";
            std::cout<<std::endl;
    }
}

//function to initialize arrays and matrices
void initialize(double A[N][N], double b[N]){
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            A[i][j] = 0.0;

    for(int j = 0; j < N; j++)
        b[j] = 0.0;
}


int main(){
    double X[N];
    double u0[N];               // initial u values (gets updated)
    double u[N];                // u values at a general time
    double b[N];                // the RHS vector
    double A[N][N];             // tri-diagonal matrix
    double D1[N], D2[N], D3[N]; // arrays to store diagonals for TDMA
    double exact[N];            // the exact solution
    double uinit[N];            // initial u values (remains fixed)

    double final_time = 0.8;    // time for which the code should run
    double dx = 1.0/(N-1);      // spatial step size
    double dt = 0.01;           // time-step
    double kappa = 0.1;         // thermal conductivity
    double alpha = kappa*dt / (2*dx*dx);

    int  n=N;
    double t = 0.0;
    int itr = 1;
    int n0 = 12;      // grid point where u variation is recorded
    char filename[] = "results/sigma.txt"; 
    remove(filename);           // remove previous version of the file

    initialize(A,b);
   
    create_grid(X, dx);
    set_initial_condition(u0);
    set_boundary_conditions(u0);

    exact_solution(exact, X, t, kappa); //get the exact solution at t=0
    write(X, u0, exact, 0); //save the initial numerical and exact solutions to a file

    update(u0, uinit);

    for(t = dt; t < final_time; t = t + dt){
        //total_time = total_time + dt; //total time elapsed
        
        create_RHS(b, u0, alpha); //construct the RHS for AX = b

        create_Tri_Diagonal_Matrix(A,alpha); //construct the coefficient matrix

        put_diagonals_in_arrays(A,D1,D2,D3); //required for the TDMA solver

        solve(D1, D2, D3, b, n); //TDMA solver, solution is stored in 'b'

        update(b,u0); //update u for next iteration

        exact_solution(exact, X, t, kappa); //get the exact solution

        write(X, u0, exact, itr); //write the numerical and exact solutions to a file

        write_at_fixed_gridpoint(b, uinit, exact[n0], t, n0); //write the solution at grid point (n0)

        itr++;
    }

    std::cout<<"solution at final step (u)"<<std::endl;
    display_vector(u0);    
    
    return 0;
}



