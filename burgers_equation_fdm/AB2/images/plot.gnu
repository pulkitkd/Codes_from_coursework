set nokey

set xlabel "Domain (x)" font ",12"
set ylabel "Solution (u(x))" font ",12"

filename(n) = sprintf("../results/solution_%d.txt", n)
plot for [i=0:5] filename(30*i) using 1:2 w lines lw 2
