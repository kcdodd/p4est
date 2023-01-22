# from libc.stdlib cimport malloc, free
from libc.string cimport memset
from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
cimport numpy as npy
import numpy as np
from mpi4py import MPI
from mpi4py.MPI cimport MPI_Comm, Comm
from p4est.utils import jagged_array
from p4est.mesh.quad import QuadMesh
from p4est.mesh.quad import quad_cell_nodes
from p4est.core._info import (
  QuadLocalInfo,
  QuadGhostInfo )
from p4est.core._adapted import QuadAdapted
from p4est.core._utils cimport (
  ndarray_from_ptr )
from p4est.core._sc cimport (
  ndarray_from_sc_array )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4estConnectivity:
  #-----------------------------------------------------------------------------
  def __init__(self, mesh):

    #...........................................................................
    if not isinstance(mesh, QuadMesh):
      raise ValueError(f"mesh must be a QuadMesh: {type(mesh)}")

    # The corners are only stored when they connect trees.
    cell_nodes = mesh.cell_nodes
    node_cells = mesh.node_cells
    node_cells_inv = mesh.node_cells_inv

    node_keep = node_cells.row_counts > 1
    num_nodes = np.count_nonzero(node_keep)

    if num_nodes < len(node_cells):
      cell_nodes = -np.ones_like(cell_nodes)
      node_cells = node_cells[node_keep]
      node_cells_inv = node_cells_inv[node_keep]

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)] = nodes

    requirements = ['C_CONTIGUOUS', 'ALIGNED', 'OWNDATA']

    self.vertices = np.require(
      mesh.verts,
      dtype = np.double,
      requirements = requirements)
    self.tree_to_vertex = np.require(
      mesh.cells,
      dtype = np.int32,
      requirements = requirements)
    self.tree_to_tree = np.require(
      mesh.cell_adj,
      dtype = np.int32,
      requirements = requirements)
    self.tree_to_face = np.require(
      mesh.cell_adj_face,
      dtype = np.int8,
      requirements = requirements)

    self.tree_to_corner = np.require(
      cell_nodes,
      dtype = np.int32,
      requirements = requirements)
    self.ctt_offset = np.require(
      node_cells.row_idx,
      dtype = np.int32,
      requirements = requirements)
    self.corner_to_tree = np.require(
      node_cells.flat,
      dtype = np.int32,
      requirements = requirements)
    self.corner_to_corner = np.require(
      node_cells_inv.flat,
      dtype = np.int8,
      requirements = requirements)

    self._init()

  #-----------------------------------------------------------------------------
  cdef _init(P4estConnectivity self):
    memset(&self._cdata, 0, sizeof(p4est_connectivity_t))

    cdef np.ndarray[double, ndim=2] vertices = self.vertices
    cdef np.ndarray[np.npy_int32, ndim=3] tree_to_vertex = self.tree_to_vertex
    cdef np.ndarray[np.npy_int32, ndim=3] tree_to_tree = self.tree_to_tree
    cdef np.ndarray[np.npy_int8, ndim=3] tree_to_face = self.tree_to_face
    cdef np.ndarray[np.npy_int32, ndim=3] tree_to_corner = self.tree_to_corner

    cdef np.ndarray[np.npy_int32, ndim=1] ctt_offset = self.ctt_offset
    cdef np.ndarray[np.npy_int32, ndim=1] corner_to_tree = self.corner_to_tree
    cdef np.ndarray[np.npy_int8, ndim=1] corner_to_corner = self.corner_to_corner

    self._cdata.num_vertices = len(vertices)
    self._cdata.vertices = <double*>vertices.data

    self._cdata.num_trees = len(tree_to_vertex)
    self._cdata.tree_to_vertex = <np.npy_int32*>(tree_to_vertex.data)
    self._cdata.tree_to_tree = <np.npy_int32*>(tree_to_tree.data)
    self._cdata.tree_to_face = <np.npy_int8*>(tree_to_face.data)

    self._cdata.num_corners = len(ctt_offset)-1
    self._cdata.ctt_offset = <np.npy_int32*>(ctt_offset.data)

    if self._cdata.num_corners > 0:
      # NOTE: intially NULL from memset
      self._cdata.tree_to_corner = <np.npy_int32*>(tree_to_corner.data)
      self._cdata.corner_to_tree = <np.npy_int32*>(corner_to_tree.data)
      self._cdata.corner_to_corner = <np.npy_int8*>(corner_to_corner.data)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:
  r"""

  Parameters
  ----------
  mesh : QuadMesh
    Mesh for the root-level cells.
  max_level : int
    (default: -1)
  comm : mpi4py.MPI.Comm
    (default: mpi4py.MPI.COMM_WORLD)

  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    mesh,
    max_level = None,
    comm = None ):

    #...........................................................................
    if not isinstance(mesh, QuadMesh):
      raise ValueError(f"mesh must be a QuadMesh: {type(mesh)}")

    if max_level is None:
      max_level = -1

    max_level = min(P4EST_MAXLEVEL, max(-1, int(max_level)))

    if comm is None:
      comm = MPI.COMM_WORLD

    #...........................................................................
    self._mesh = mesh
    self._max_level = max_level
    self._comm = comm

    self._local = QuadLocalInfo(0)
    self._ghost = jagged_array(
      data = QuadGhostInfo(0),
      row_idx = np.array([0], dtype = np.int32) )

    self._mirror = jagged_array(
      data = self._local,
      row_idx = np.array([0], dtype = np.int32) )

    self._connectivity = P4estConnectivity(self._mesh)

    self._init()

  #-----------------------------------------------------------------------------
  cdef _init(P4est self):

    cdef p4est_t* p4est = NULL
    cdef sc_MPI_Comm comm = <sc_MPI_Comm> (<Comm>self._comm).ob_mpi
    cdef p4est_connectivity_t* connectivity = &(self._connectivity._cdata)

    with nogil:
      p4est = p4est_new_ext(
        comm,
        connectivity,
        0,
        0,
        0,
        sizeof(aux_quadrant_data_t),
        <p4est_init_t>_init_quadrant,
        <void*>self )

    self._p4est = p4est
    self._sync_info()

  #-----------------------------------------------------------------------------
  def __dealloc__(self):
    """Deallocate c-level system
    """
    self.free()

  #-----------------------------------------------------------------------------
  def free(self):
    if self._p4est == NULL:
      return

    p4est_destroy(self._p4est)
    self._p4est = NULL
    self._mesh = None

  #-----------------------------------------------------------------------------
  def __enter__(self):
    return self

  #-----------------------------------------------------------------------------
  def __exit__(self, type, value, traceback):
    self.free()

    return False

  #-----------------------------------------------------------------------------
  def coord(self,
    offset,
    where = None ):

    if offset is None:
      offset = 0.5*np.ones((2,), dtype = np.float64)

    else:
      offset = np.asarray(offset, dtype = np.float64)

    if where is None:
      where = slice(None)

    info = self._local[where]

    # compute the local discrete width of the leaf within the root cell
    # NOTE: the root level = 0 is 2**P4EST_MAXLEVEL wide, and all refinement
    # levels are smaller by factors that are powers of 2.
    qwidth = np.left_shift(1, P4EST_MAXLEVEL - info.level.astype(np.int32))

    leaf_offset = ( info.origin + offset[None,:] * qwidth[:,None] ) / P4EST_ROOT_LEN

    return self._mesh.coord(
      offset = leaf_offset,
      where = info.root )

  #-----------------------------------------------------------------------------
  def _adapt(self):

    _set_leaf_adapt(
      trees = <p4est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      adapt = self._local._adapt )

    with nogil:
      p4est_refine_ext(
        self._p4est,
        # recursive
        0,
        self._max_level,
        <p4est_refine_t>_refine_quadrant,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

      p4est_coarsen_ext(
        self._p4est,
        # recursive
        0,
        # orphans
        1,
        <p4est_coarsen_t>_coarsen_quadrants,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

      p4est_balance_ext(
        self._p4est,
        P4EST_CONNECT_FULL,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

    return self._sync_info()

  #-----------------------------------------------------------------------------
  def _partition(self):

    rank = self._comm.rank
    init_info = self._local
    init_idx = np.copy(ndarray_from_ptr(
      write = False,
      dtype = np.int64,
      count = self._comm.size+1,
      arr = <char*>self._p4est.global_first_quadrant))

    _set_leaf_weight(
      trees = <p4est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      weight = self._local._weight )

    with nogil:
      p4est_partition_ext(
        self._p4est,
        # partition_for_coarsening
        0,
        <p4est_weight_t>_weight_quadrant )

    moved, refined, coarsened = self._sync_info()

    assert len(refined) == 0
    assert len(coarsened) == 0

    final_idx = np.copy(ndarray_from_ptr(
      write = False,
      dtype = np.int64,
      count = self._comm.size+1,
      arr = <char*>self._p4est.global_first_quadrant))

    tx0 = init_idx[rank]
    tx1 = init_idx[rank+1]
    tx_idx = np.clip(final_idx - tx0, 0, tx1 - tx0)

    tx = jagged_array(data = init_info, row_idx = tx_idx)

    rx0 = final_idx[rank]
    rx1 = final_idx[rank+1]
    rx_idx = np.clip(init_idx - rx0, 0, rx1 - rx0)

    rx = jagged_array(data = self._local, row_idx = rx_idx)

    return tx, rx

  #-----------------------------------------------------------------------------
  cdef _sync_info(P4est self):
    cdef:
      p4est_ghost_t* ghost
      p4est_mesh_t* mesh
      p4est_nodes_t* nodes

      npy.npy_int32 num_moved = 0
      npy.npy_int32 num_adapted = 0

    with nogil:
      ghost = p4est_ghost_new(
        self._p4est,
        P4EST_CONNECT_FULL)

      mesh = p4est_mesh_new_ext(
        self._p4est,
        ghost,
        # compute_tree_index
        0,
        # compute_level_lists
        0,
        P4EST_CONNECT_FULL)

      nodes = p4est_nodes_new(
        self._p4est,
        NULL)

    prev_info = self._local
    self._local = QuadLocalInfo(mesh.local_num_quadrants)
    ghost_flat = QuadGhostInfo(mesh.ghost_num_quadrants)

    _count_leaf_changes(
      rank = self._comm.rank,
      trees = <p4est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      num_moved = &num_moved,
      num_adapted = &num_adapted )

    leaf_moved_from = -np.ones(
      (num_moved,),
      dtype = np.int32)

    leaf_moved_to = -np.ones(
      (num_moved,),
      dtype = np.int32)

    leaf_adapted = np.zeros(
      (num_adapted,),
      dtype = np.int8)

    leaf_adapted_coarse = -np.ones(
      (num_adapted,),
      dtype = np.int32)

    leaf_adapted_fine = -np.ones(
      (num_adapted, 2, 2),
      dtype = np.int32 )

    _sync_local_info(
      rank = self._comm.rank,
      trees = <p4est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      quad_to_quad = ndarray_from_ptr(
        write = False,
        dtype = np.int32,
        count = 4*mesh.local_num_quadrants,
        arr = <char*>mesh.quad_to_quad).reshape(-1,2,2),
      quad_to_face = ndarray_from_ptr(
        write = False,
        dtype = np.int8,
        count = 4*mesh.local_num_quadrants,
        arr = <char*>mesh.quad_to_face).reshape(-1,2,2),
      quad_to_half = ndarray_from_sc_array(
        write = False,
        dtype = np.int32,
        subitems = 2,
        arr = mesh.quad_to_half).reshape(-1,2),
      # out
      root = self._local.root,
      level = self._local.level,
      origin = self._local.origin,
      weight = self._local.weight,
      adapt = self._local.adapt,
      cell_adj = self._local.cell_adj,
      cell_adj_face = self._local.cell_adj_face,
      cell_adj_subface = self._local.cell_adj_subface,
      cell_adj_order = self._local.cell_adj_order,
      cell_adj_level = self._local.cell_adj_level,
      leaf_moved_from = leaf_moved_from,
      leaf_moved_to = leaf_moved_to,
      leaf_adapted = leaf_adapted,
      leaf_adapted_fine = leaf_adapted_fine,
      leaf_adapted_coarse = leaf_adapted_coarse )

    self._local.idx = np.arange(mesh.local_num_quadrants)
    self.local.cell_nodes = ndarray_from_ptr(
        write = False,
        dtype = np.int32,
        count = 4 * mesh.local_num_quadrants,
        arr = <char *>nodes.local_nodes).reshape(-1,2,2)
    vnodes = np.arange(nodes.indep_nodes.elem_count + nodes.face_hangings.elem_count)
    vert_nodes, self.local.cell_nodes, node_cells, node_cells_inv = quad_cell_nodes(self.local.cell_nodes, vnodes)

    ranks = np.concatenate([
      np.full(
        (len(self._local),),
        fill_value = self._comm.rank,
        dtype = np.int32),
      ndarray_from_ptr(
        write = False,
        dtype = np.intc,
        count = mesh.ghost_num_quadrants,
        arr = <char*>mesh.ghost_to_proc)])

    self._local.cell_adj_rank = ranks[self._local.cell_adj]

    ghost_proc_offsets = np.copy(ndarray_from_ptr(
      write = False,
      dtype = np.int32,
      count = ghost.mpisize + 1,
      arr = <char*>ghost.proc_offsets))

    _sync_ghost_info(
      ghosts = <p4est_quadrant_t*> ghost.ghosts.array,
      num_ghosts = mesh.ghost_num_quadrants,
      proc_offsets = ghost_proc_offsets,
      # out
      rank = ghost_flat.rank,
      root = ghost_flat.root,
      idx = ghost_flat.idx,
      level = ghost_flat.level,
      origin = ghost_flat.origin )

    self._ghost = jagged_array(
      data = ghost_flat,
      row_idx = ghost_proc_offsets )

    # indices of each mirror in leaf_info
    mirrors_idx = np.zeros((ghost.mirrors.elem_count,), dtype = np.int32)

    _sync_mirror_idx(
      mirrors = <p4est_quadrant_t*> ghost.mirrors.array,
      num_mirrors = ghost.mirrors.elem_count,
      mirrors_idx = mirrors_idx )

    # indices into mirror_proc_mirrors for each rank
    mirror_proc_offsets = np.copy(ndarray_from_ptr(
      write = False,
      dtype = np.int32,
      count = ghost.mpisize + 1,
      arr = <char*>ghost.mirror_proc_offsets))

    # indices into mirrors (grouped by processor rank)
    mirror_proc_mirrors = ndarray_from_ptr(
      write = False,
      dtype = np.int32,
      # NOTE: a mirror can be repeated for difference ranks, so need to use
      # offsets to get total number of rank-wise mirrors (with repeats)
      count = mirror_proc_offsets[-1],
      arr = <char*>ghost.mirror_proc_mirrors)

    self._mirror = jagged_array(
      data = mirrors_idx[mirror_proc_mirrors],
      row_idx = mirror_proc_offsets )

    p4est_nodes_destroy(nodes)
    p4est_mesh_destroy(mesh)
    p4est_ghost_destroy(ghost)

    refined_mask = leaf_adapted > 0
    coarsened_mask = ~refined_mask

    fine_idx = leaf_adapted_fine[refined_mask]
    refined_idx = leaf_adapted_coarse[refined_mask]

    coarse_idx = leaf_adapted_coarse[coarsened_mask]
    coarsened_idx = leaf_adapted_fine[coarsened_mask]

    moved = QuadAdapted(
      dst = self._local[leaf_moved_to],
      src = prev_info[leaf_moved_from] )

    refined = QuadAdapted(
      dst = self._local[fine_idx],
      src = prev_info[refined_idx] )

    coarsened = QuadAdapted(
      dst = self._local[coarse_idx],
      src = prev_info[coarsened_idx] )

    return moved, refined, coarsened

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _set_leaf_adapt(
  p4est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  npy.npy_int8[:] adapt ) nogil:

  cdef:
    p4est_tree_t* tree = NULL
    p4est_quadrant_t* quads = NULL
    p4est_quadrant_t* cell = NULL
    aux_quadrant_data_t* cell_aux

    npy.npy_int32 root_idx = 0
    npy.npy_int32 q = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p4est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.idx = tree.quadrants_offset + q
      cell_aux.adapt = adapt[cell_aux.idx]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _set_leaf_weight(
  p4est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  npy.npy_int32[:] weight ) nogil:

  cdef:
    p4est_tree_t* tree = NULL
    p4est_quadrant_t* quads = NULL
    p4est_quadrant_t* cell = NULL
    aux_quadrant_data_t* cell_aux

    npy.npy_int32 root_idx = 0
    npy.npy_int32 q = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p4est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.idx = tree.quadrants_offset + q
      cell_aux.weight = weight[cell_aux.idx]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _count_leaf_changes(
  npy.npy_int32 rank,
  const p4est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  npy.npy_int32* num_moved,
  npy.npy_int32* num_adapted ) nogil:

  cdef:
    const p4est_tree_t* tree = NULL
    const p4est_quadrant_t* quads = NULL
    const p4est_quadrant_t* cell = NULL
    const aux_quadrant_data_t* cell_aux

    npy.npy_int32 root_idx = 0
    npy.npy_int32 q = 0
    npy.npy_int32 refine_idx = 0
    npy.npy_int32 coarse_idx = 0
    npy.npy_int32 move_idx = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p4est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data

      if cell_aux.rank == rank:
        if cell_aux.adapted == 1:
          refine_idx += 1
        elif cell_aux.adapted == -1:
          coarse_idx += 1
        else:
          move_idx += 1

  num_moved[0] = move_idx
  num_adapted[0] = coarse_idx + refine_idx // 4

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _sync_local_info(
  npy.npy_int32 rank,
  p4est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  # npy.npy_int32 num_local,
  # npy.npy_int32 num_ghost,
  # const int[:] ghost_to_proc,
  const npy.npy_int32[:,:,:] quad_to_quad,
  const npy.npy_int8[:,:,:] quad_to_face,
  const npy.npy_int32[:,:] quad_to_half,
  # out
  npy.npy_int32[:] root,
  npy.npy_int8[:] level,
  npy.npy_int32[:,:] origin,
  npy.npy_int32[:] weight,
  npy.npy_int8[:] adapt,
  npy.npy_int32[:,:,:,:] cell_adj,
  npy.npy_int8[:,:,:] cell_adj_face,
  npy.npy_int8[:,:,:] cell_adj_subface,
  npy.npy_int8[:,:,:] cell_adj_order,
  npy.npy_int8[:,:,:] cell_adj_level,
  npy.npy_int32[:] leaf_moved_from,
  npy.npy_int32[:] leaf_moved_to,
  npy.npy_int8[:] leaf_adapted,
  npy.npy_int32[:,:,:] leaf_adapted_fine,
  npy.npy_int32[:] leaf_adapted_coarse ) nogil:
  """Low-level method to sync quadrant data with the contiguous arrays.
  """

  cdef:
    p4est_tree_t* tree = NULL
    p4est_quadrant_t* quads = NULL
    p4est_quadrant_t* cell = NULL
    aux_quadrant_data_t* cell_aux

    npy.npy_int32 i = 0
    npy.npy_int32 j = 0
    npy.npy_int32 k = 0
    npy.npy_int32 q = 0

    npy.npy_int32 root_idx = 0
    npy.npy_int32 adapt_idx = 0
    npy.npy_int32 move_idx = 0
    npy.npy_int8 max_prev_adapt = 0

    npy.npy_int32 cell_idx = 0

    npy.npy_int32 prev_cell_idx = 0
    npy.npy_int32[:,:] prev_fine_idx

    npy.npy_int32 cell_adj_idx = 0
    npy.npy_int8 cell_adj_face_idx = 0
    npy.npy_int8 face_order = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p4est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data

      cell_idx = tree.quadrants_offset + q

      # finish tracking of changes due to refine/coarsen operation
      # now that the indices are known, and thus able to map previous quadrants
      if cell_aux.adapted == 1:
        # refined
        # which quadrant of the replaced cell
        k = (
          (cell_aux.replaced_idx[1] >= 0)
          + 2*(cell_aux.replaced_idx[2] >= 0)
          + 3*(cell_aux.replaced_idx[3] >= 0) )

        j = k // 2
        i = k % 2

        leaf_adapted_fine[adapt_idx, j, i] = cell_idx
        # print(f"{cell_idx}, adapt= {adapt_idx}, refine: {i},{j}")

        if k == 3:
          leaf_adapted[adapt_idx] = 1
          prev_cell_idx = cell_aux.replaced_idx[3]
          leaf_adapted_coarse[adapt_idx] = prev_cell_idx
          # print(f"{cell_idx}, adapt= {adapt_idx}, replaced: {prev_cell_idx}")
          adapt_idx += 1

      elif cell_aux.adapted == -1:
        # coarsened
        leaf_adapted[adapt_idx] = -1
        leaf_adapted_coarse[adapt_idx] = cell_idx

        prev_fine_idx = leaf_adapted_fine[adapt_idx]
        prev_fine_idx[0,0] = cell_aux.replaced_idx[0]
        prev_fine_idx[0,1] = cell_aux.replaced_idx[1]
        prev_fine_idx[1,0] = cell_aux.replaced_idx[2]
        prev_fine_idx[1,1] = cell_aux.replaced_idx[3]

        # print(f"{cell_idx}, adapt= {adapt_idx}, coarse: {prev_fine_idx}")
        adapt_idx += 1

      elif cell_aux.rank == rank:
        prev_cell_idx = cell_aux.idx
        leaf_moved_from[move_idx] = prev_cell_idx
        leaf_moved_to[move_idx] = cell_idx
        # print(f"{cell_idx}, move= {move_idx}: {prev_cell_idx}")
        move_idx += 1

      # reset auxiliary information
      cell_aux.rank = rank
      cell_aux.idx = cell_idx
      cell_aux.adapted = 0

      adapt[cell_idx] = cell_aux.adapt
      weight[cell_idx] = cell_aux.weight

      root[cell_idx] = root_idx
      level[cell_idx] = cell.level
      origin[cell_idx,0] = cell.x
      origin[cell_idx,1] = cell.y

      # get adjacency information from the mesh
      for i in range(2):
        for j in  range(2):

          cell_adj_idx = quad_to_quad[cell_idx,i,j]
          cell_adj_face_idx = quad_to_face[cell_idx,i,j]

          # A value of v = 0..7 indicates one same-size neighbor.
          # A value of v = 8..23 indicates a double-size neighbor.
          # designates the subface of the large neighbor that the quadrant
          # touches (this is the same as the large neighbor's face corner).
          # A value of v = -8..-1 indicates two half-size neighbors.
          face_order = cell_adj_face_idx % 8

          cell_adj_face[cell_idx,i,j] = face_order % 4
          cell_adj_order[cell_idx,i,j] = face_order // 4

          cell_adj_subface[cell_idx,i,j] = cell_adj_face_idx // 16


          if cell_adj_face_idx >= 0:
            cell_adj[cell_idx,i,j] = cell_adj_idx
            cell_adj_level[cell_idx,i,j] = 0 if (cell_adj_face_idx < 8) else -1
          else:
            # In this case the corresponding quad_to_quad index points into the
            # quad_to_half array that stores two quadrant numbers per index,
            cell_adj[cell_idx,i,j] = quad_to_half[cell_adj_idx]
            cell_adj_level[cell_idx,i,j] = 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _sync_ghost_info(
  p4est_quadrant_t* ghosts,
  npy.npy_int32 num_ghosts,
  const npy.npy_int32[:] proc_offsets,
  # out
  npy.npy_int32[:] rank,
  npy.npy_int32[:] root,
  npy.npy_int32[:] idx,
  npy.npy_int8[:] level,
  npy.npy_int32[:,:] origin ) nogil:


  cdef:
    p4est_quadrant_t* cell = NULL
    npy.npy_int32 p = 0
    npy.npy_int32 q = 0

  for p in range(len(proc_offsets)-1):
    for q in range(proc_offsets[p], proc_offsets[p+1]):
      cell = &ghosts[q]

      rank[q] = p
      root[q] = cell.p.piggy3.which_tree
      idx[q] = cell.p.piggy3.local_num
      level[q] = cell.level
      origin[q,0] = cell.x
      origin[q,1] = cell.y

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _sync_mirror_idx(
  p4est_quadrant_t* mirrors,
  npy.npy_int32 num_mirrors,
  # out
  npy.npy_int32[:] mirrors_idx ) nogil:


  cdef:
    p4est_quadrant_t* cell = NULL
    npy.npy_int32 q = 0

  for q in range(num_mirrors):
    cell = &mirrors[q]
    mirrors_idx[q] = cell.p.piggy3.local_num

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _init_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) nogil:

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data
  cell_aux.rank = -1
  cell_aux.idx = -1
  cell_aux.adapt = 0
  cell_aux.weight = 1

  cell_aux.adapted = 0
  cell_aux.replaced_idx[0] = -1
  cell_aux.replaced_idx[1] = -1
  cell_aux.replaced_idx[2] = -1
  cell_aux.replaced_idx[3] = -1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _replace_quadrants(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  int num_outgoing,
  p4est_quadrant_t* outgoing[],
  int num_incoming,
  p4est_quadrant_t* incoming[] ) nogil:

  cdef:
    p4est_quadrant_t* cell
    aux_quadrant_data_t* cell_aux

    p4est_quadrant_t* _cell
    aux_quadrant_data_t* _cell_aux

    npy.npy_int8 prev_adapt
    npy.npy_int8 prev_weight

    npy.npy_int8 k

  # NOTE: incoming means the 'added' quadrants, and outgoing means 'removed'
  if num_outgoing == 4:
    # Coarsening: remove 4 -> add 1
    # assert num_outgoing == 4
    # assert num_incoming == 1

    cell = incoming[0]

    # flag that this index currently refers to the adapt array
    cell_aux = <aux_quadrant_data_t*>cell.p.user_data
    cell_aux.adapted = -1

    prev_adapt = -128
    prev_weight = 0

    for k in range(4):
      _cell = outgoing[k]
      _cell_aux = <aux_quadrant_data_t*>_cell.p.user_data

      prev_adapt = max(prev_adapt, _cell_aux.adapt)
      prev_weight = max(prev_weight, _cell_aux.weight)

      cell_aux.replaced_idx[k] = _cell_aux.idx

    cell_aux.rank = _cell_aux.rank
    cell_aux.adapt = prev_adapt + 1
    cell_aux.weight = prev_weight

  else:
    # Refining: remove 1 -> add 4
    # assert num_outgoing == 1
    # assert num_incoming == 4

    _cell = outgoing[0]
    _cell_aux = <aux_quadrant_data_t*>_cell.p.user_data

    prev_adapt = _cell_aux.adapt
    prev_weight = _cell_aux.weight

    for k in range(4):
      cell = incoming[k]

      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.rank = _cell_aux.rank
      cell_aux.adapt = prev_adapt - 1
      cell_aux.weight = prev_weight

      cell_aux.adapted = 1
      cell_aux.replaced_idx[k] = _cell_aux.idx
      cell_aux.replaced_idx[(k+1)%4] = -1
      cell_aux.replaced_idx[(k+2)%4] = -1
      cell_aux.replaced_idx[(k+3)%4] = -1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _refine_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) nogil:

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return cell_aux.adapt > 0

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _coarsen_quadrants(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrants[] ) nogil:

  cdef aux_quadrant_data_t* cell_aux

  for k in range(4):
    cell_aux = <aux_quadrant_data_t*>quadrants[k].p.user_data

    if cell_aux.adapt >= 0:
      return 0

  return 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _weight_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) nogil:

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return cell_aux.weight

