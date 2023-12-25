fn=$1
scp -P 44 qazal@72.220.147.45:/home/qazal/tinygrad/prg ./"tests/$fn.c" && scp -P 44 qazal@72.220.147.45:/home/qazal/tinygrad/0 ./"tests/$fn.s"
