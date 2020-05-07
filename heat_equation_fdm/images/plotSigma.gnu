set logscale y

set key top right

set xlabel "Time (t)"
set ylabel "Solution u(x0)"

plot '../results/sigma.txt' u 1:2 w p pt 7 lc rgb "blue" lw 2 t "u(x0) (numerical)" , '../results/sigma.txt' u 1:3 w l lc rgb "red" lw 2 t "u(x0) (exact)"
