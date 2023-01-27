cimport numpy as np
# from mpi4py.MPI cimport MPI_Comm, Comm
from p4est.core._sc cimport (
  sc_MPI_Comm,
  sc_array_t,
  sc_mempool_t )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: Some important definitions are not in p4est.h
cdef extern from "p4est_extended.h" nogil:
  const int P4EST_MAXLEVEL
  const int P4EST_ROOT_LEN

  #.............................................................................
  ctypedef np.npy_int32 p4est_topidx_t
  ctypedef np.npy_int32 p4est_qcoord_t
  ctypedef np.npy_int32 p4est_locidx_t
  ctypedef np.npy_int64 p4est_gloidx_t
  ctypedef np.npy_uint64 p4est_lid_t


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

  #-----------------------------------------------------------------------------
  ctypedef enum p4est_connect_type_t:
    P4EST_CONNECT_FACE = 21,
    P4EST_CONNECT_CORNER = 22,
    P4EST_CONNECT_FULL = P4EST_CONNECT_CORNER

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

  #.............................................................................
  ctypedef struct p4est_ghost_t:
    int mpisize
    p4est_topidx_t num_trees
    p4est_connect_type_t btype

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
  ctypedef struct p4est_mesh_t:

    p4est_locidx_t      local_num_quadrants
    p4est_locidx_t      ghost_num_quadrants

    p4est_topidx_t     *quad_to_tree
    int                *ghost_to_proc

    p4est_locidx_t     *quad_to_quad
    np.npy_int8        *quad_to_face
    sc_array_t         *quad_to_half
    sc_array_t         *quad_level

    p4est_locidx_t      local_num_corners
    p4est_locidx_t     *quad_to_corner
    sc_array_t         *corner_offset
    sc_array_t         *corner_quad
    sc_array_t         *corner_corner

  #.............................................................................
  ctypedef struct p4est_mesh_face_neighbor_t:
    p4est_t            *p4est
    p4est_ghost_t      *ghost
    p4est_mesh_t       *mesh
    p4est_nodes_t      *nodes

    p4est_topidx_t      which_tree
    p4est_locidx_t      quadrant_id
    p4est_locidx_t      quadrant_code

    int                 face
    int                 subface

    p4est_locidx_t      current_qtq

  #-----------------------------------------------------------------------------
  # Callback function prototype to initialize the quadrant's user data.
  ctypedef void (*p4est_init_t)(
    p4est_t* p4est,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t* quadrant )

  # Callback function prototype to replace one set of quadrants with another.
  # If the mesh is being refined, num_outgoing will be 1 and num_incoming will
  # be 4, and vice versa if the mesh is being coarsened.
  ctypedef void (*p4est_replace_t) (
    p4est_t * p4est,
    p4est_topidx_t which_tree,
    int num_outgoing,
    p4est_quadrant_t * outgoing[],
    int num_incoming,
    p4est_quadrant_t * incoming[])

  # Callback function prototype to decide for refinement.
  ctypedef int (*p4est_refine_t)(
    p4est_t * p4est,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t * quadrant)

  # Callback function prototype to decide for coarsening.
  ctypedef int (*p4est_coarsen_t)(
    p4est_t * p4est,
    p4est_topidx_t cell_idx,
    # Pointers to 4 siblings in Morton ordering.
    p4est_quadrant_t * quadrants[])

  # Callback function prototype to calculate weights for partitioning.
  # NOTE: Global sum of weights must fit into a 64bit integer.
  ctypedef int (*p4est_weight_t)(
    p4est_t * p4est,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t * quadrant)

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
  # Make a deep copy of a p4est
  p4est_t* p4est_copy(
    p4est_t* input,
    int copy_data )

  #.............................................................................
  void p4est_destroy(p4est_t* p4est)

  #-----------------------------------------------------------------------------
  p4est_ghost_t* p4est_ghost_new(
    p4est_t* p4est,
    p4est_connect_type_t btype)

  #.............................................................................
  void p4est_ghost_destroy(
    p4est_ghost_t* ghost)

  #.............................................................................
  p4est_mesh_t* p4est_mesh_new_ext(
    p4est_t* p4est,
    p4est_ghost_t* ghost,
    int compute_tree_index,
    int compute_level_lists,
    p4est_connect_type_t btype)

  #.............................................................................
  void p4est_mesh_destroy(p4est_mesh_t * mesh)

  #.............................................................................
  p4est_nodes_t* p4est_nodes_new (
    p4est_t * p4est,
    p4est_ghost_t * ghost)

  #.............................................................................
  void p4est_nodes_destroy(p4est_nodes_t*)

  #.............................................................................
  void p4est_refine_ext(
    p4est_t * p4est,
    int refine_recursive,
    int maxlevel,
    p4est_refine_t refine_fn,
    p4est_init_t init_fn,
    p4est_replace_t replace_fn )

  #.............................................................................
  void p4est_coarsen_ext(
    p4est_t * p4est,
    int coarsen_recursive,
    int callback_orphans,
    p4est_coarsen_t coarsen_fn,
    p4est_init_t init_fn,
    p4est_replace_t replace_fn )

  #.............................................................................
  # 2:1 balance the size differences of neighboring elements in a forest.
  void p4est_balance_ext(
    p4est_t * p4est,
    p4est_connect_type_t btype,
    p4est_init_t init_fn,
    p4est_replace_t replace_fn )

  void p4est_balance_subtree_ext(
    p4est_t * p4est,
    p4est_connect_type_t btype,
    p4est_topidx_t which_tree,
    p4est_init_t init_fn,
    p4est_replace_t replace_fn )

  #.............................................................................
  # Equally partition the forest.
  # The partition can be by element count or by a user-defined weight.
  #
  # The forest will be partitioned between processors such that they
  # have an approximately equal number of quadrants (or sum of weights).
  #
  # On one process, the function noops and does not call the weight callback.
  # Otherwise, the weight callback is called once per quadrant in order.
  p4est_gloidx_t p4est_partition_ext(
    p4est_t * p4est,
    int allow_for_coarsening,
    p4est_weight_t weight_fn )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "p4est_nodes.h" nogil:
  #.............................................................................
  ctypedef struct p4est_nodes_t:

    p4est_locidx_t num_local_quadrants
    p4est_locidx_t num_owned_indeps, num_owned_shared
    p4est_locidx_t offset_owned_indeps
    sc_array_t indep_nodes
    sc_array_t face_hangings
    p4est_locidx_t *local_nodes
    sc_array_t shared_indeps
    p4est_locidx_t *shared_offsets
    int *nonlocal_ranks
    p4est_locidx_t *global_owned_indeps


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef extern from "p4est_iterate.h" nogil:
  #.............................................................................
  ctypedef struct p4est_ghost_t:
    int mpisize
    p4est_topidx_t num_trees
    # which neighbors are in the ghost layer
    p4est_connect_type_t btype

    # An array of quadrants which make up the ghost layer around \a
    # p4est.  Their piggy3 data member is filled with their owner's tree
    # and local number (cumulative over trees).  Quadrants are ordered in \c
    # p4est_quadrant_compare_piggy order.  These are quadrants inside the
    # neighboring tree, i.e., \c p4est_quadrant_is_inside() is true for the
    # quadrant and the neighboring tree.

    sc_array_t ghosts
    p4est_locidx_t *tree_offsets
    p4est_locidx_t *proc_offsets

    # An array of local quadrants that touch the parallel boundary from the
    # inside, i.e., that are ghosts in the perspective of at least one other
    # processor.  The storage convention is the same as for \c ghosts above.

    sc_array_t mirrors
    p4est_locidx_t *mirror_tree_offsets
    p4est_locidx_t *mirror_proc_mirrors
    p4est_locidx_t *mirror_proc_offsets

    p4est_locidx_t *mirror_proc_fronts
    p4est_locidx_t *mirror_proc_front_offsets

  #.............................................................................
  ctypedef struct p4est_iter_volume_info_t:
    p4est_t *p4est
    p4est_ghost_t *ghost_layer
    p4est_quadrant_t *quad
    p4est_locidx_t quadid
    p4est_topidx_t treeid

  # The prototype for a function that p4est_iterate will execute at every
  # quadrant local to the current process.
  ctypedef void (*p4est_iter_volume_t) (
    p4est_iter_volume_info_t * info,
    void *user_data )

  #.............................................................................
  ctypedef struct p4est_iter_face_info_t:
    p4est_t *p4est
    p4est_ghost_t *ghost_layer
    np.npy_int8 orientation
    np.npy_int8 tree_boundary
    # array of p4est_iter_face_side_t type
    sc_array_t sides

  # The prototype for a function that p4est_iterate will execute wherever two
  # quadrants share a face: the face can be a 2:1 hanging face, it does not have
  # to be conformal.
  ctypedef void (*p4est_iter_face_t) (
    p4est_iter_face_info_t * info,
    void *user_data )

  #.............................................................................
  ctypedef struct p4est_iter_corner_info_t:
    p4est_t *p4est
    p4est_ghost_t *ghost_layer
    np.npy_int8 tree_boundary
    # array of p4est_iter_corner_side_t type
    sc_array_t sides

  # The prototype for a function that p4est_iterate will execute wherever two
  # quadrants share a face: the face can be a 2:1 hanging face, it does not have
  # to be conformal.
  ctypedef void (*p4est_iter_corner_t) (
    p4est_iter_corner_info_t * info,
    void *user_data )

  #.............................................................................
  void p4est_iterate(
    p4est_t* p4est,
    p4est_ghost_t * ghost_layer,
    void *user_data,
    p4est_iter_volume_t iter_volume,
    p4est_iter_face_t iter_face,
    p4est_iter_corner_t iter_corner)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ctypedef struct aux_quadrant_data_t:
  np.npy_int32 rank
  np.npy_int32 idx
  np.npy_int32 weight

  np.npy_int32 coarse_idx
  np.npy_int32 fine_idx[4]

  np.npy_int8 adapt
  np.npy_int8 adapted
  np.npy_int8 replacement
  np.npy_int8 _future_flag1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4estConnectivity:
  cdef vertices
  cdef tree_to_vertex
  cdef tree_to_tree
  cdef tree_to_face
  cdef tree_to_corner

  cdef ctt_offset
  cdef corner_to_tree
  cdef corner_to_corner

  cdef p4est_connectivity_t _cdata

  #-----------------------------------------------------------------------------
  cdef _init(P4estConnectivity self)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:

  cdef public _mesh
  cdef public _comm
  cdef public np.npy_int8 _max_level

  cdef public _local
  cdef public _ghost
  cdef public _mirror

  cdef P4estConnectivity _connectivity
  cdef p4est_t* _p4est

  #-----------------------------------------------------------------------------
  cdef _init(P4est self)

  # #-----------------------------------------------------------------------------
  # cdef void _adapt(P4est self) nogil

  # #-----------------------------------------------------------------------------
  # cdef void _partition(P4est self) nogil

  #-----------------------------------------------------------------------------
  cdef _sync_info(P4est self)
