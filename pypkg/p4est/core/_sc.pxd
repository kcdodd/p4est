from libc.stdio cimport FILE
cimport numpy as np

# from mpi4py.MPI cimport MPI_Comm, Comm

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "sc.h" nogil:
  #.............................................................................
  ctypedef int sc_MPI_Comm

  #.............................................................................
  ctypedef void (*sc_log_handler_t)(
    FILE * log_stream,
    const char *filename,
    int lineno,
    int package,
    int category,
    int priority,
    const char *msg)



  #-----------------------------------------------------------------------------
  void sc_set_log_defaults(
    FILE * log_stream,
    sc_log_handler_t log_handler,
    int log_threshold)

  #-----------------------------------------------------------------------------
  void sc_init(
    sc_MPI_Comm mpicomm,
    int catch_signals,
    int print_backtrace,
    sc_log_handler_t log_handler,
    int log_threshold)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "sc_containers.h" nogil:
  #.............................................................................
  ctypedef struct sc_array_t:
    # interface variables
    # size of a single element
    size_t elem_size
    # number of valid elements
    size_t elem_count

    # implementation variables
    # number of allocated bytes
    # or -(number of viewed bytes + 1)
    # if this is a view: the "+ 1"
    # distinguishes an array of size 0
    # from a view of size 0
    ssize_t byte_alloc

    # linear array to store elements
    char *array

  #.............................................................................
  ctypedef struct sc_mempool_t:
    # interface variables
    #size of a single element
    size_t elem_size
    #number of valid elements
    size_t elem_count
    #Boolean is set in constructor.
    int zero_and_persist

    # implementation variables
    #ifdef SC_MEMPOOL_MSTAMP
    # sc_mstamp_t mstamp
    #our own obstack replacement
    #else
    # struct obstack obstack
    #holds the allocated elements
    #endif
    sc_array_t freed
    #buffers the freed elements

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef ndarray_from_sc_array(write, dtype, subitems, sc_array_t* arr)