
from cpython cimport PyObject
cimport numpy as np

from mpi4py.MPI cimport MPI_Comm, Comm


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "p4est.h":
  const int P4EST_MAXLEVEL

  #.............................................................................
  ctypedef int sc_MPI_Comm

  #.............................................................................
  ctypedef np.npy_int32 p4est_topidx_t
  ctypedef np.npy_int32 p4est_qcoord_t
  ctypedef np.npy_int32 p4est_locidx_t
  ctypedef np.npy_int64 p4est_gloidx_t

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

  #.............................................................................
  ctypedef struct p4est_inspect_t:
    # Use sc_ranges to determine the asymmetric communication pattern.
    # If \a use_balance_ranges is false (the default), sc_notify is used.
    int use_balance_ranges
    # If true, call both sc_ranges and sc_notify and verify consistency.
    # Which is actually used is still determined by \a use_balance_ranges.
    int use_balance_ranges_notify
    # Verify sc_ranges and/or sc_notify as applicable.
    int use_balance_verify
    # If positive and smaller than p4est_num ranges, overrides it
    int balance_max_ranges
    size_t balance_A_count_in
    size_t balance_A_count_out
    size_t balance_comm_sent
    size_t balance_comm_nzpeers
    size_t balance_B_count_in
    size_t balance_B_count_out
    size_t balance_zero_sends[2]
    size_t balance_zero_receives[2]
    double balance_A
    double balance_comm
    double balance_B
    double balance_ranges
    #time spent in sc_ranges
    double balance_notify
    #time spent in sc_notify
    #* time spent in sc_notify_allgather
    double balance_notify_allgather
    int use_B

  #.............................................................................
  ctypedef struct p4est_connectivity_t:
    # the number of vertices that define the \a embedding of the forest
    # (not the topology)
    p4est_topidx_t num_vertices
    # the number of trees
    p4est_topidx_t num_trees
    # the number of corners that help define topology
    p4est_topidx_t num_corners
    # an array of size (3 * \a num_vertices)
    double *vertices
    # embed each tree into \f$R^3\f$ for e.g. visualization (see p4est_vtk.h)
    p4est_topidx_t *tree_to_vertex

    # bytes per tree in tree_to_attr
    size_t tree_attr_bytes
    # not touched by p4est
    char *tree_to_attr

    # (4 * \a num_trees) neighbors across faces
    p4est_topidx_t *tree_to_tree
    # (4 * \a num_trees) face to face+orientation (see description)
    np.npy_int8 *tree_to_face

    # (4 * \a num_trees) or NULL (see description)
    p4est_topidx_t *tree_to_corner
    # corner to offset in \a corner_to_tree and \a corner_to_corner
    p4est_topidx_t *ctt_offset
    # list of trees that meet at a corner
    p4est_topidx_t *corner_to_tree
    # list of tree-corners that meet at a corner
    np.npy_int8 *corner_to_corner

  #.............................................................................
  # of ghost octants, store the tree and owner rank
  ctypedef struct p4est_quadrant_piggy1:
    p4est_topidx_t which_tree
    int owner_rank

  #.............................................................................
  # of transformed octants, store the original tree and the target tree
  ctypedef struct p4est_quadrant_piggy2:
    p4est_topidx_t which_tree
    p4est_topidx_t from_tree

  #.............................................................................
  # of ghost octants, store the tree and index in the owner's numbering
  ctypedef struct p4est_quadrant_piggy3:
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

  #.............................................................................
  ctypedef struct p4est_tree_t:
    # locally stored quadrants
    sc_array_t quadrants
    # first local descendant
    p4est_quadrant_t first_desc
    # last local descendant
    p4est_quadrant_t last_desc
    # cumulative sum over earlier trees on this processor (locals only)
    p4est_locidx_t quadrants_offset
    # locals only
    p4est_locidx_t quadrants_per_level[P4EST_MAXLEVEL + 1]
    # highest local quadrant level
    np.npy_int8 maxlevel

  #.............................................................................
  ctypedef struct p4est_t:
    sc_MPI_Comm mpicomm
    # MPI communicator
    # number of MPI processes
    int mpisize
    # this process's MPI rank
    int mpirank
    # flag if communicator is owned
    int mpicomm_owned
    # size of per-quadrant p.user_data
    # (see p4est_quadrant_t::p4est_quadrant_data::user_data)
    size_t data_size
    # convenience pointer for users, never touched by p4est
    void* user_pointer

    # Gets bumped on mesh change
    long revision

    # 0-based index of first local tree, must be -1 for an emptyprocessor
    p4est_topidx_t first_local_tree

    # 0-based index of last local tree, must be -2 for an empty processor
    p4est_topidx_t last_local_tree

    # number of quadrants on all trees on this processor
    p4est_locidx_t local_num_quadrants

    # number of quadrants on all trees on all processors
    p4est_gloidx_t global_num_quadrants

    # first global quadrant index for each process and 1 beyond
    p4est_gloidx_t* global_first_quadrant

    # first smallest possible quad for each process and 1 beyond
    p4est_quadrant_t* global_first_position

    # connectivity structure, not owned
    p4est_connectivity_t* connectivity
    # array of all trees
    sc_array_t* trees

    # memory allocator for user data
    # WARNING: This is NULL if data size equals zero.
    sc_mempool_t* user_data_pool

    # memory allocator for temporary quadrants
    sc_mempool_t* quadrant_pool

    # algorithmic switches
    p4est_inspect_t* inspect

  #-----------------------------------------------------------------------------
  ctypedef void (*p4est_init_t)(
    p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant )

  #-----------------------------------------------------------------------------
  p4est_t* p4est_new_ext(
    sc_MPI_Comm mpicomm,
    p4est_connectivity_t* connectivity,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform,
    size_t data_size,
    p4est_init_t init_fn,
    void* user_pointer )

  #-----------------------------------------------------------------------------
  void p4est_destroy(p4est_t* p4est)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _init_quadrant(
  p4est_t* p4est,
  p4est_topidx_t which_tree,
  p4est_quadrant_t* quadrant )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:
  cdef _comm
  cdef p4est_t* _p4est
  cdef p4est_connectivity_t _tmap
  cdef np.ndarray _verts
  cdef np.ndarray _cells
  cdef np.ndarray _cell_adj
  cdef np.ndarray _cell_adj_face

  cdef np.ndarray _tree_to_corner
  cdef np.ndarray _corner_to_tree_offset
  cdef np.ndarray _corner_to_tree
  cdef np.ndarray _corner_to_corner

  #-----------------------------------------------------------------------------
  cdef _init_c_data(
    P4est self,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform )

  #-----------------------------------------------------------------------------
  cdef _init_quadrant(
    P4est self,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant )