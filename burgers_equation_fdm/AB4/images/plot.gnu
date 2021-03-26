set nokey

set xlabel "Domain (x)" font ",12"
set ylabel "Solution (u(x))" font ",12"

set style line 1 lt 1 lc rgb '#0072bd' lw 2 # blue
set style line 2 lt 1 lc rgb '#d95319' lw 2 # orange
set style line 3 lt 1 lc rgb '#edb120' lw 2 # yellow
set style line 4 lt 1 lc rgb '#7e2f8e' lw 2 # purple
set style line 5 lt 1 lc rgb '#77ac30' lw 2 # green
#set style line 16 lt 1 lc rgb '#4dbeee' # light-blue
#set style line 17 lt 1 lc rgb '#a2142f' # red


filename(n) = sprintf("../results/solution_%d.txt", n)
plot for [i=0:5] filename(30*i) using 1:2 w lines ls i+1
