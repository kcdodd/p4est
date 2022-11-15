from cpython cimport PyObject
cimport numpy as np

from mpi4py.MPI cimport MPI_Comm, Comm


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "p4est.h":
  #.............................................................................
  ctypedef np.npy_int32 p4est_topidx_t
  ctypedef np.npy_int32 p4est_qcoord_t
  ctypedef np.npy_int32 p4est_locidx_t
  ctypedef np.npy_int64 p4est_gloidx_t

  #.............................................................................
  # of ghost octants, store the tree and owner rank
  cdef struct p4est_quadrant_piggy1:
      p4est_topidx_t which_tree
      int owner_rank

  #.............................................................................
  # of transformed octants, store the original tree and the target tree
  cdef struct p4est_quadrant_piggy2:
      p4est_topidx_t which_tree
      p4est_topidx_t from_tree

  #.............................................................................
  # of ghost octants, store the tree and index in the owner's numbering
  cdef struct p4est_quadrant_piggy3:
      p4est_topidx_t which_tree
      p4est_locidx_t local_num

  #.............................................................................
  # union of additional data attached to a quadrant
  cdef union p4est_quadrant_data:
    # user data never changed by p4est 
    void* user_data
    long user_long
    int user_int

    # the tree containing the quadrant
    # (used in auxiliary octants such
    # as the ghost octants in
    # p4est_ghost_t) 
    p4est_topidx_t which_tree

    p4est_quadrant_piggy1 piggy1
    p4est_quadrant_piggy2 piggy2
    p4est_quadrant_piggy3 piggy3

  #.............................................................................
  ctypedef struct p4est_quadrant_t:
    
    # coordinates
    p4est_qcoord_t x 
    p4est_qcoord_t y

    # level of refinement
    np.npy_int8 level

    # padding
    np.npy_int8 pad8
    np.npy_int16 pad16

    p4est_quadrant_data p