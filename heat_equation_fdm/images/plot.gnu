set key top right

set xlabel "Domain (x)" font ",12"
set ylabel "Solution (u(x))" font ",12"

filename(n) = sprintf("../results/solution_%d.txt", n)
plot for [i=0:10] filename(4*i) using 2:4 t (i==0? 'Exact':'') with lines lc rgb "red" lw 2
replot for [i=0:10] filename(4*i) using 2:3 t (i==0? 'Numerical':'') w points lc rgb "blue" lw 0.5 pt 7

