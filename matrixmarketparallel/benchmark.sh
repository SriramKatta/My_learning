#!/bin/bash -l
#
#SBATCH -J fast_mmio
#SBATCH -p singlenode
#SBATCH --time=3:00:00
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV


module load intel intelmpi likwid cmake

for np in $(seq 1 72 )
do
    data=$(likwid-mpirun -n $np ./build/fast_mmio ./matrix/af_shell10/af_shell10.mtx \
    | grep -i "time taken")

    echo "$np $data"
done