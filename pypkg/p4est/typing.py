from typing import (
  Optional,
  Union,
  Literal,
  TypeVar,
  NewType )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#: A variable size
N = TypeVar('N', bound = int)

#: A variable size
M = TypeVar('M', bound = int)

#: A variable number of vertices
NV = TypeVar('NV', bound = int)

#: A variable number of nodes
NN = TypeVar('NN', bound = int)

#: A variable number of edges
NE = TypeVar('NE', bound = int)

#: A variable number of cells
NC = TypeVar('NC', bound = int)

#: A variable number of processes in MPI.Comm
NP = TypeVar('NP', bound = int)