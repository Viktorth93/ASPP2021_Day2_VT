
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD


rank = np.zeros(1)
rankSum = np.zeros(1)

rank[0] = comm.Get_rank()

comm.Reduce(rank, rankSum, op=MPI.SUM, root=0)


if comm.rank == 0:
    print("The sum of all ranks is", rankSum)


