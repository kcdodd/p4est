cimport numpy as np
from p4est.core._sc cimport (
  sc_MPI_Comm,
  sc_array_t,
  sc_mempool_t )
from p4est.core._info cimport (
  HexLocalInfo,
  HexGhostInfo )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: Some important definitions are not in p8est.h
cdef extern from "p8est_extended.h" nogil:
  const int P8EST_MAXLEVEL
  const int P8EST_ROOT_LEN

  #.............................................................................
  ctypedef np.npy_int32 p4est_topidx_t
  ctypedef np.npy_int32 p4est_qcoord_t
  ctypedef np.npy_int32 p4est_locidx_t
  ctypedef np.npy_int64 p4est_gloidx_t
  ctypedef np.npy_uint64 p4est_lid_t

  #.............................................................................
  ctypedef struct p8est_inspect_t:
    # Use sc_ranges to determine the asymmetric communication pattern.
    # If \a use_balance_ranges is false (the default), sc_notify is used.
    int use_balance_ranges
    # If true, call both sc_ranges and sc_notify and verify consistency.
    # Which is actually used is still determined by \a use_balance_ranges.
    int use_balance_ranges_notify
    # Verify sc_ranges and/or sc_notify as applicable.
    int use_balance_verify
    # If positive and smaller than p8est_num ranges, overrides it
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
  ctypedef struct p8est_connectivity_t:
    # the number of vertices that define the \a embedding of the forest
    # (not the topology)
    p4est_topidx_t num_vertices
    # the number of trees
    p4est_topidx_t num_trees

    # the number of edges that help define the topology
    p4est_topidx_t num_edges
    # the number of corners that help define topology
    p4est_topidx_t num_corners

    # an array of size (3 * num_vertices)
    double* vertices
    # embed each tree into \f$R^3\f$ for e.g. visualization (see p8est_vtk.h)
    p4est_topidx_t* tree_to_vertex

    # bytes per tree in tree_to_attr
    size_t tree_attr_bytes
    # not touched by p4est
    char* tree_to_attr

    # (6 * num_trees) neighbors across faces
    p4est_topidx_t* tree_to_tree
    # (6 * num_trees) face to face+orientation
    np.npy_int8* tree_to_face
    # 12 * num_trees) or NULL
    p4est_topidx_t* tree_to_edge
    # (8 * num_trees) or NULL (see description)
    p4est_topidx_t* tree_to_corner

    p4est_topidx_t* ett_offset
    # list of trees that meet at an edge
    p4est_topidx_t* edge_to_tree
    # tree-edges+orientations
    np.npy_int8* edge_to_edge

    # corner to offset in corner_to_tree and corner_to_corner
    p4est_topidx_t* ctt_offset
    # list of trees that meet at a corner
    p4est_topidx_t* corner_to_tree
    # list of tree-corners that meet at a corner
    np.npy_int8* corner_to_corner

  #-----------------------------------------------------------------------------
  ctypedef enum p8est_connect_type_t:
    P8EST_CONNECT_SELF = 30,
    P8EST_CONNECT_FACE = 31,
    P8EST_CONNECT_EDGE = 32,
    P8EST_CONNECT_CORNER = 33,
    P8EST_CONNECT_FULL = P8EST_CONNECT_CORNER

  #.............................................................................
  # of ghost octants, store the tree and owner rank
  ctypedef struct p8est_quadrant_piggy1:
    p4est_topidx_t which_tree
    int owner_rank

  #.............................................................................
  # of transformed octants, store the original tree and the target tree
  ctypedef struct p8est_quadrant_piggy2:
    p4est_topidx_t which_tree
    p4est_topidx_t from_tree

  #.............................................................................
  # of ghost octants, store the tree and index in the owner's numbering
  ctypedef struct p8est_quadrant_piggy3:
    p4est_topidx_t which_tree
    p4est_locidx_t local_num

  #.............................................................................
  # union of additional data attached to a quadrant
  cdef union p8est_quadrant_data:
    # user data never changed by p4est
    void* user_data
    long user_long
    int user_int

    # the tree containing the quadrant
    # (used in auxiliary octants such
    # as the ghost octants in
    # p8est_ghost_t)
    p4est_topidx_t which_tree

    p8est_quadrant_piggy1 piggy1
    p8est_quadrant_piggy2 piggy2
    p8est_quadrant_piggy3 piggy3

  #.............................................................................
  ctypedef struct p8est_quadrant_t:

    # coordinates
    p4est_qcoord_t x
    p4est_qcoord_t y
    p4est_qcoord_t z

    # level of refinement
    np.npy_int8 level

    # padding
    np.npy_int8 pad8
    np.npy_int16 pad16

    p8est_quadrant_data p

  #.............................................................................
  ctypedef struct p8est_tree_t:
    # locally stored quadrants
    sc_array_t quadrants
    # first local descendant
    p8est_quadrant_t first_desc
    # last local descendant
    p8est_quadrant_t last_desc
    # cumulative sum over earlier trees on this processor (locals only)
    p4est_locidx_t quadrants_offset
    # locals only
    p4est_locidx_t quadrants_per_level[P8EST_MAXLEVEL + 1]
    # highest local quadrant level
    np.npy_int8 maxlevel

  #.............................................................................
  ctypedef struct p8est_t:
    sc_MPI_Comm mpicomm
    # MPI communicator
    # number of MPI processes
    int mpisize
    # this process's MPI rank
    int mpirank
    # flag if communicator is owned
    int mpicomm_owned
    # size of per-quadrant p.user_data
    # (see p8est_quadrant_t::p8est_quadrant_data::user_data)
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
    p8est_quadrant_t* global_first_position

    # connectivity structure, not owned
    p8est_connectivity_t* connectivity
    # array of all trees
    sc_array_t* trees

    # memory allocator for user data
    # WARNING: This is NULL if data size equals zero.
    sc_mempool_t* user_data_pool

    # memory allocator for temporary quadrants
    sc_mempool_t* quadrant_pool

    # algorithmic switches
    p8est_inspect_t* inspect

  #.............................................................................
  ctypedef struct p8est_ghost_t:
    int mpisize
    p4est_topidx_t num_trees
    p8est_connect_type_t btype

    sc_array_t ghosts
    p4est_locidx_t *tree_offsets
    p4est_locidx_t *proc_offsets

    sc_array_t mirrors
    p4est_locidx_t *mirror_tree_offsets
    p4est_locidx_t *mirror_proc_mirrors
    p4est_locidx_t *mirror_proc_offsets

    p4est_locidx_t *mirror_proc_fronts
    p4est_locidx_t *mirror_proc_front_offsets

  #.............................................................................
  ctypedef struct p8est_mesh_t:

    p4est_locidx_t local_num_quadrants
    p4est_locidx_t ghost_num_quadrants

    p4est_topidx_t *quad_to_tree
    int *ghost_to_proc

    p4est_locidx_t *quad_to_quad
    np.npy_int8 *quad_to_face
    sc_array_t *quad_to_half
    sc_array_t *quad_level

    p4est_locidx_t local_num_edges
    p4est_locidx_t *quad_to_edge
    sc_array_t *edge_offset
    sc_array_t *edge_quad
    sc_array_t *edge_edge

    p4est_locidx_t local_num_corners
    p4est_locidx_t *quad_to_corner
    sc_array_t *corner_offset
    sc_array_t *corner_quad
    sc_array_t *corner_corner

  #.............................................................................
  ctypedef struct p8est_mesh_face_neighbor_t:
    p8est_t *p4est
    p8est_ghost_t *ghost
    p8est_mesh_t *mesh

    p4est_topidx_t which_tree
    p4est_locidx_t quadrant_id
    p4est_locidx_t quadrant_code

    int face
    int subface

    p4est_locidx_t current_qtq

  #-----------------------------------------------------------------------------
  # Callback function prototype to initialize the quadrant's user data.
  ctypedef void (*p8est_init_t)(
    p8est_t* p4est,
    p4est_topidx_t cell_idx,
    p8est_quadrant_t* quadrant )

  # Callback function prototype to replace one set of quadrants with another.
  # If the mesh is being refined, num_outgoing will be 1 and num_incoming will
  # be 4, and vice versa if the mesh is being coarsened.
  ctypedef void (*p8est_replace_t) (
    p8est_t * p4est,
    p4est_topidx_t which_tree,
    int num_outgoing,
    p8est_quadrant_t * outgoing[],
    int num_incoming,
    p8est_quadrant_t * incoming[])

  # Callback function prototype to decide for refinement.
  ctypedef int (*p8est_refine_t)(
    p8est_t * p4est,
    p4est_topidx_t cell_idx,
    p8est_quadrant_t * quadrant)

  # Callback function prototype to decide for coarsening.
  ctypedef int (*p8est_coarsen_t)(
    p8est_t * p4est,
    p4est_topidx_t cell_idx,
    # Pointers to 4 siblings in Morton ordering.
    p8est_quadrant_t * quadrants[])

  # Callback function prototype to calculate weights for partitioning.
  # NOTE: Global sum of weights must fit into a 64bit integer.
  ctypedef int (*p8est_weight_t)(
    p8est_t * p4est,
    p4est_topidx_t cell_idx,
    p8est_quadrant_t * quadrant)

  #-----------------------------------------------------------------------------
  p8est_t* p8est_new_ext(
    sc_MPI_Comm mpicomm,
    p8est_connectivity_t* connectivity,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform,
    size_t data_size,
    p8est_init_t init_fn,
    void* user_pointer )

  #.............................................................................
  void p8est_destroy(p8est_t* p4est)

  #-----------------------------------------------------------------------------
  p8est_ghost_t* p8est_ghost_new(
    p8est_t* p4est,
    p8est_connect_type_t btype)

  #.............................................................................
  void p8est_ghost_destroy(
    p8est_ghost_t* ghost)

  #.............................................................................
  p8est_mesh_t* p8est_mesh_new_ext(
    p8est_t* p4est,
    p8est_ghost_t* ghost,
    int compute_tree_index,
    int compute_level_lists,
    p8est_connect_type_t btype)

  #.............................................................................
  void p8est_mesh_destroy(p8est_mesh_t * mesh)

  #.............................................................................
  void p8est_refine_ext(
    p8est_t * p4est,
    int refine_recursive,
    int maxlevel,
    p8est_refine_t refine_fn,
    p8est_init_t init_fn,
    p8est_replace_t replace_fn )

  #.............................................................................
  void p8est_coarsen_ext(
    p8est_t * p4est,
    int coarsen_recursive,
    int callback_orphans,
    p8est_coarsen_t coarsen_fn,
    p8est_init_t init_fn,
    p8est_replace_t replace_fn )

  #.............................................................................
  # 2:1 balance the size differences of neighboring elements in a forest.
  void p8est_balance_ext(
    p8est_t * p4est,
    p8est_connect_type_t btype,
    p8est_init_t init_fn,
    p8est_replace_t replace_fn )

  void p8est_balance_subtree_ext(
    p8est_t * p4est,
    p8est_connect_type_t btype,
    p4est_topidx_t which_tree,
    p8est_init_t init_fn,
    p8est_replace_t replace_fn )

  #.............................................................................
  # Equally partition the forest.
  # The partition can be by element count or by a user-defined weight.
  #
  # The forest will be partitioned between processors such that they
  # have an approximately equal number of quadrants (or sum of weights).
  #
  # On one process, the function noops and does not call the weight callback.
  # Otherwise, the weight callback is called once per quadrant in order.
  p4est_gloidx_t p8est_partition_ext(
    p8est_t * p4est,
    int allow_for_coarsening,
    p8est_weight_t weight_fn )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ctypedef struct aux_quadrant_data_t:
  np.npy_int32 idx
  np.npy_int8 adapt
  np.npy_int32 weight

  np.npy_int8 adapted
  np.npy_int8 _future_flag1
  np.npy_int8 _future_flag2

  np.npy_int32 replaced_idx[8]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P8estConnectivity:
  cdef vertices
  cdef tree_to_vertex
  cdef tree_to_tree
  cdef tree_to_face
  cdef tree_to_edge
  cdef tree_to_corner

  cdef ett_offset
  cdef edge_to_tree
  cdef edge_to_edge

  cdef ctt_offset
  cdef corner_to_tree
  cdef corner_to_corner

  cdef p8est_connectivity_t _cdata

  #-----------------------------------------------------------------------------
  cdef _init(P8estConnectivity self)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P8est:

  cdef public _mesh
  cdef public _comm
  cdef public np.npy_int8 _max_level

  cdef public HexLocalInfo _local
  cdef public _ghost
  cdef public _mirror

  cdef P8estConnectivity _connectivity
  cdef p8est_t* _p4est

  #-----------------------------------------------------------------------------
  cdef _init(P8est self)

  #-----------------------------------------------------------------------------
  cdef void _adapt(P8est self) nogil

  #-----------------------------------------------------------------------------
  cdef void _partition(P8est self) nogil

  #-----------------------------------------------------------------------------
  cdef _sync_local(P8est self)
