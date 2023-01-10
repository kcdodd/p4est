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
from p4est.mesh.hex import (
  HexMeshBase,
  HexMesh )
from p4est.core._leaf_info import (
  HexInfo,
  HexGhostInfo )
from p4est.core._adapted import HexAdapted
from p4est.core._utils cimport (
  ndarray_from_ptr )
from p4est.core._sc cimport (
  ndarray_from_sc_array )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P8est:
  """

  Parameters
  ----------
  mesh : HexMesh
  min_level : None | int
    (default: 0)
  max_level : None | int
    (default: -1)
  comm : None | mpi4py.MPI.Comm
    (default: mpi4py.MPI.COMM_WORLD)

  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    mesh,
    min_level = None,
    max_level = None,
    comm = None ):

    #...........................................................................
    if not isinstance(mesh, HexMeshBase):
      raise ValueError(f"mesh must be a HexMeshBase: {type(mesh)}")

    if min_level is None:
      min_level = 0

    min_level = min(127, max(0, int(min_level)))

    if max_level is None:
      max_level = -1

    max_level = min(127, max(-1, int(max_level)))

    if comm is None:
      comm = MPI.COMM_WORLD


    #...........................................................................
    self._max_level = P8EST_MAXLEVEL
    self._comm = comm
    self._mesh = mesh
    self._min_level = min_level

    self._leaf_info = HexInfo(0)
    self._ghost_info = HexGhostInfo(0)
    self._rank_ghosts = None
    self._rank_mirrors = None

    self._init()

  #-----------------------------------------------------------------------------
  cdef _init(P8est self):

    memset(&self._connectivity, 0, sizeof(p8est_connectivity_t))

    cdef np.ndarray[double, ndim=2] vertices = self._mesh.verts
    cdef np.ndarray[np.npy_int32, ndim=4] tree_to_vertex = self._mesh.cells
    cdef np.ndarray[np.npy_int32, ndim=3] tree_to_tree = self._mesh.cell_adj
    cdef np.ndarray[np.npy_int8, ndim=3] tree_to_face = self._mesh.cell_adj_face
    cdef np.ndarray[np.npy_int32, ndim=4] tree_to_edge = self._mesh.cell_edges
    cdef np.ndarray[np.npy_int32, ndim=4] tree_to_corner = self._mesh.cell_nodes

    cdef np.ndarray[np.npy_int32, ndim=1] ett_offset = np.ascontiguousarray(self._mesh.edge_cells.row_idx)
    cdef np.ndarray[np.npy_int32, ndim=1] edge_to_tree = np.ascontiguousarray(self._mesh.edge_cells.flat)
    cdef np.ndarray[np.npy_int8, ndim=1] edge_to_edge = np.ascontiguousarray(self._mesh.edge_cells_inv.flat)

    cdef np.ndarray[np.npy_int32, ndim=1] ctt_offset = np.ascontiguousarray(self._mesh.node_cells.row_idx)
    cdef np.ndarray[np.npy_int32, ndim=1] corner_to_tree = np.ascontiguousarray(self._mesh.node_cells.flat)
    cdef np.ndarray[np.npy_int8, ndim=1] corner_to_corner = np.ascontiguousarray(self._mesh.node_cells_inv.flat)

    self._connectivity.num_vertices = len(vertices)
    self._connectivity.vertices = <double*>vertices.data

    self._connectivity.num_trees = len(tree_to_vertex)
    self._connectivity.tree_to_vertex = <np.npy_int32*>(tree_to_vertex.data)
    self._connectivity.tree_to_tree = <np.npy_int32*>(tree_to_tree.data)
    self._connectivity.tree_to_face = <np.npy_int8*>(tree_to_face.data)

    self._connectivity.num_edges = len(ett_offset)-1
    self._connectivity.ett_offset = <np.npy_int32*>(ett_offset.data)

    if self._connectivity.num_edges > 0:
      # NOTE: intially NULL from memset
      self._connectivity.tree_to_edge = <np.npy_int32*>(tree_to_edge.data)
      self._connectivity.edge_to_tree = <np.npy_int32*>(edge_to_tree.data)
      self._connectivity.edge_to_edge = <np.npy_int8*>(edge_to_edge.data)

    self._connectivity.num_corners = len(ctt_offset)-1
    self._connectivity.ctt_offset = <np.npy_int32*>(ctt_offset.data)

    if self._connectivity.num_corners > 0:
      # NOTE: intially NULL from memset
      self._connectivity.tree_to_corner = <np.npy_int32*>(tree_to_corner.data)
      self._connectivity.corner_to_tree = <np.npy_int32*>(corner_to_tree.data)
      self._connectivity.corner_to_corner = <np.npy_int8*>(corner_to_corner.data)


    cdef p8est_t* p4est = NULL
    cdef sc_MPI_Comm comm = <sc_MPI_Comm> (<Comm>self.comm).ob_mpi
    cdef p8est_connectivity_t* connectivity = &(self._connectivity)

    with nogil:
      p4est = p8est_new_ext(
        comm,
        connectivity,
        0,
        self._min_level,
        0,
        sizeof(aux_quadrant_data_t),
        <p8est_init_t>_init_quadrant,
        <void*>self )

    self._p4est = p4est
    self._sync_leaf_info()

  #-----------------------------------------------------------------------------
  def __dealloc__(self):
    """Deallocate c-level system
    """
    self.free()

  #-----------------------------------------------------------------------------
  def free(self):
    if self._p4est == NULL:
      return

    p8est_destroy(self._p4est)
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
  @property
  def max_level( self ):
    return self._max_level

  #-----------------------------------------------------------------------------
  @property
  def comm( self ):
    return self._comm

  #-----------------------------------------------------------------------------
  @property
  def mesh( self ):
    return self._mesh

  #-----------------------------------------------------------------------------
  @property
  def shape( self ):
    return self._shape

  #-----------------------------------------------------------------------------
  @property
  def dtype( self ):
    return self._dtype

  #-----------------------------------------------------------------------------
  @property
  def leaf_info(self):
    return self._leaf_info

  #-----------------------------------------------------------------------------
  @property
  def ghost_info(self):
    return self._ghost_info

  #-----------------------------------------------------------------------------
  @property
  def rank_ghosts(self):
    return self._rank_ghosts

  #-----------------------------------------------------------------------------
  @property
  def rank_mirrors(self):
    return self._rank_mirrors

  #-----------------------------------------------------------------------------
  def leaf_coord(self,
    uv = None,
    idx = None,
    interp = None ):
    r"""

    Parameters
    ----------
    uv : None | array of shape = (2,)
      Relative position within each leaf to compute the coordinates, normalized
      to the range [0.0, 1.0] along each edge of the leaf.
      (default: (0.0, 0.0))
    idx : None | any single-dimension ndarray index
      Computes coordinates only for this subset of leaves. (default: slice(None))
    interp : None | callable
      Interpolation function ()

    Returns
    -------
    coords: array of shape = (len(leaf_info), 3)
      Absolute coordinates of the given relative position in each leaf
    """

    if uv is None:
      uv = 0.5*np.ones((3,), dtype = np.float64)

    else:
      uv = np.asarray(uv, dtype = np.float64)

    if idx is None:
      idx = slice(None)

    info = self._leaf_info[idx]

    # compute the local discrete width of the leaf within the root cell
    # NOTE: the root level = 0 is 2**P8EST_MAXLEVEL wide, and all refinement
    # levels are smaller by factors that are powers of 2.
    qwidth = np.left_shift(1, P8EST_MAXLEVEL - info.level.astype(np.int32))

    cell_verts = self.mesh.verts[ self.mesh.cells[ info.root ] ]

    # compute the relative position within the root cell
    UV = np.clip(
      ( info.origin + uv[None,:] * qwidth[:,None] ) / P8EST_ROOT_LEN,
      0.0, 1.0)

    # Compute the coefficients for bilinear interpolation of the root cells
    # vertices absolute position onto the desired position relative to the leaf.

    if interp is None:
      _UV = 1.0 - UV

      # tri-linear interpolation
      c = np.empty(cell_verts.shape[:4])
      c[:,0,0,0] = _UV[:,0]*_UV[:,1]*_UV[:,2]
      c[:,0,0,1] = UV[:,0]*_UV[:,1]*_UV[:,2]
      c[:,0,1,0] = _UV[:,0]*UV[:,1]*_UV[:,2]
      c[:,0,1,1] = UV[:,0]*UV[:,1]*_UV[:,2]

      c[:,1,0,0] = _UV[:,0]*_UV[:,1]*UV[:,2]
      c[:,1,0,1] = UV[:,0]*_UV[:,1]*UV[:,2]
      c[:,1,1,0] = _UV[:,0]*UV[:,1]*UV[:,2]
      c[:,1,1,1] = UV[:,0]*UV[:,1]*UV[:,2]

      # Perform the interpolation
      return np.sum(c[:,:,:,:,None] * cell_verts, axis = (1,2,3))

    else:
      return interp(cell_verts, UV)

  #-----------------------------------------------------------------------------
  def adapt(self):
    """Applies refinement, coarsening, and then balances based on ``leaf_info.adapt``.

    Returns
    -------
    refined : HexAdapted
    coarsened : HexAdapted
    """

    _set_leaf_adapt(
      trees = <p8est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      adapt = self._leaf_info.adapt )

    with nogil:
      self._adapt()

    return self._sync_leaf_info()

  #-----------------------------------------------------------------------------
  cdef void _adapt(P8est self) nogil:

    p8est_refine_ext(
      self._p4est,
      # recursive
      0,
      self._max_level,
      <p8est_refine_t>_refine_quadrant,
      <p8est_init_t>_init_quadrant,
      <p8est_replace_t> _replace_quadrants )

    p8est_coarsen_ext(
      self._p4est,
      # recursive
      0,
      # orphans
      1,
      <p8est_coarsen_t>_coarsen_quadrants,
      <p8est_init_t>_init_quadrant,
      <p8est_replace_t> _replace_quadrants )

    p8est_balance_ext(
      self._p4est,
      P8EST_CONNECT_FULL,
      <p8est_init_t>_init_quadrant,
      <p8est_replace_t> _replace_quadrants )

  #-----------------------------------------------------------------------------
  cdef void _partition(P8est self) nogil:

    with nogil:
      p8est_partition_ext(
        self._p4est,
        1,
        <p8est_weight_t>_weight_quadrant )

  #-----------------------------------------------------------------------------
  cdef _sync_leaf_info(P8est self):
    cdef:
      p8est_ghost_t* ghost
      p8est_mesh_t* mesh

    with nogil:
      ghost = p8est_ghost_new(
        self._p4est,
        P8EST_CONNECT_FULL)

      mesh = p8est_mesh_new_ext(
        self._p4est,
        ghost,
        # compute_tree_index
        0,
        # compute_level_lists
        0,
        P8EST_CONNECT_FULL)

    prev_leaf_info = self._leaf_info
    self._leaf_info = HexInfo(mesh.local_num_quadrants)
    self._ghost_info = HexGhostInfo(mesh.ghost_num_quadrants)

    num_adapted = _count_leaf_adapted(
      trees = <p8est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree )

    leaf_adapted = np.zeros(
      (num_adapted,),
      dtype = np.int8)

    leaf_adapted_coarse = -np.ones(
      (num_adapted,),
      dtype = np.int32)

    leaf_adapted_fine = -np.ones(
      (num_adapted, 2, 2, 2),
      dtype = np.int32 )

    _sync_leaf_info(
      trees = <p8est_tree_t*>self._p4est.trees.array,
      first_local_tree = self._p4est.first_local_tree,
      last_local_tree = self._p4est.last_local_tree,
      quad_to_quad = ndarray_from_ptr(
        write = False,
        dtype = np.int32,
        count = 6*mesh.local_num_quadrants,
        arr = <char*>mesh.quad_to_quad).reshape(-1,3,2),
      quad_to_face = ndarray_from_ptr(
        write = False,
        dtype = np.int8,
        count = 6*mesh.local_num_quadrants,
        arr = <char*>mesh.quad_to_face).reshape(-1,3,2),
      quad_to_half = ndarray_from_sc_array(
        write = False,
        dtype = np.int32,
        subitems = 4,
        arr = mesh.quad_to_half).reshape(-1,2,2),
      # out
      root = self._leaf_info.root,
      level = self._leaf_info.level,
      origin = self._leaf_info.origin,
      weight = self._leaf_info.weight,
      adapt = self._leaf_info.adapt,
      cell_adj = self._leaf_info.cell_adj,
      cell_adj_face = self._leaf_info.cell_adj_face,
      cell_adj_subface = self._leaf_info.cell_adj_subface,
      cell_adj_order = self._leaf_info.cell_adj_order,
      cell_adj_level = self._leaf_info.cell_adj_level,
      leaf_adapted = leaf_adapted,
      leaf_adapted_fine = leaf_adapted_fine,
      leaf_adapted_coarse = leaf_adapted_coarse )

    self._leaf_info.idx = np.arange(mesh.local_num_quadrants)

    ranks = np.concatenate([
      np.full(
        (len(self._leaf_info),),
        fill_value = self.comm.rank,
        dtype = np.int32),
      ndarray_from_ptr(
        write = False,
        dtype = np.intc,
        count = mesh.ghost_num_quadrants,
        arr = <char*>mesh.ghost_to_proc)])

    self._leaf_info.cell_adj_rank = ranks[self._leaf_info.cell_adj]

    # translate cell_adj indicies for ghost quadrants to {-ng..-1)
    # NOTE: This can be used to flag ghost vs local based on the sign,
    # but also, negative indexing can be used (without modification) for indexing
    # ghost data either appended to the end of an array of local data, or stored
    # in a separate array of only ghost data.
    nl = mesh.local_num_quadrants
    ng = mesh.ghost_num_quadrants
    cell_adj = self._leaf_info.cell_adj

    self._leaf_info.cell_adj -= (ng + nl) * (cell_adj // nl)

    ghost_proc_offsets = np.copy(ndarray_from_ptr(
      write = False,
      dtype = np.int32,
      count = ghost.mpisize + 1,
      arr = <char*>ghost.proc_offsets))

    _sync_ghost_info(
      ghosts = <p8est_quadrant_t*> ghost.ghosts.array,
      num_ghosts = mesh.ghost_num_quadrants,
      proc_offsets = ghost_proc_offsets,
      # out
      rank = self._ghost_info.rank,
      root = self._ghost_info.root,
      idx = self._ghost_info.idx,
      level = self._ghost_info.level,
      origin = self._ghost_info.origin )

    self._rank_ghosts = jagged_array(
      data = self._ghost_info,
      row_idx = ghost_proc_offsets )

    # indices of each mirror in leaf_info
    mirrors_idx = np.zeros((ghost.mirrors.elem_count,), dtype = np.int32)

    _sync_mirror_idx(
      mirrors = <p8est_quadrant_t*> ghost.mirrors.array,
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

    mirror_info = self._leaf_info[mirrors_idx[mirror_proc_mirrors]]

    self._rank_mirrors = jagged_array(
      data = mirror_info,
      row_idx = mirror_proc_offsets )

    p8est_mesh_destroy(mesh)
    p8est_ghost_destroy(ghost)

    refined_mask = leaf_adapted > 0
    coarsened_mask = ~refined_mask

    fine_idx = leaf_adapted_fine[refined_mask]
    refined_idx = leaf_adapted_coarse[refined_mask]

    coarse_idx = leaf_adapted_coarse[coarsened_mask]
    coarsened_idx = leaf_adapted_fine[coarsened_mask]

    refined = HexAdapted(
      idx = fine_idx,
      info = self._leaf_info[fine_idx],
      replaced_idx = refined_idx,
      replaced_info = prev_leaf_info[refined_idx] )

    coarsened = HexAdapted(
      idx = coarse_idx,
      info = self._leaf_info[coarse_idx],
      replaced_idx = coarsened_idx,
      replaced_info = prev_leaf_info[coarsened_idx] )

    return refined, coarsened

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef void _set_leaf_adapt(
  p8est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  npy.npy_int8[:] adapt ) nogil:

  cdef:
    p8est_tree_t* tree = NULL
    p8est_quadrant_t* quads = NULL
    p8est_quadrant_t* cell = NULL
    aux_quadrant_data_t* cell_aux

    npy.npy_int32 root_idx = 0
    npy.npy_int32 q = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p8est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.idx = tree.quadrants_offset + q
      cell_aux.adapt = adapt[cell_aux.idx]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef npy.npy_int32 _count_leaf_adapted(
  const p8est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree ) nogil:

  cdef:
    const p8est_tree_t* tree = NULL
    const p8est_quadrant_t* quads = NULL
    const p8est_quadrant_t* cell = NULL
    const aux_quadrant_data_t* cell_aux

    npy.npy_int32 root_idx = 0
    npy.npy_int32 q = 0
    npy.npy_int32 adapt_idx = 0


  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p8est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data

      if cell_aux.adapted != 0:
        adapt_idx += 1

  return adapt_idx

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef _sync_leaf_info(
  p8est_tree_t* trees,
  npy.npy_int32 first_local_tree,
  npy.npy_int32 last_local_tree,
  # npy.npy_int32 num_local,
  # npy.npy_int32 num_ghost,
  # const int[:] ghost_to_proc,
  const npy.npy_int32[:,:,:] quad_to_quad,
  const npy.npy_int8[:,:,:] quad_to_face,
  const npy.npy_int32[:,:,:] quad_to_half,
  # out
  npy.npy_int32[:] root,
  npy.npy_int8[:] level,
  npy.npy_int32[:,:] origin,
  npy.npy_int32[:] weight,
  npy.npy_int8[:] adapt,
  npy.npy_int32[:,:,:,:,:] cell_adj,
  npy.npy_int8[:,:,:] cell_adj_face,
  npy.npy_int8[:,:,:] cell_adj_subface,
  npy.npy_int8[:,:,:] cell_adj_order,
  npy.npy_int8[:,:,:] cell_adj_level,
  npy.npy_int8[:] leaf_adapted,
  npy.npy_int32[:,:,:,:] leaf_adapted_fine,
  npy.npy_int32[:] leaf_adapted_coarse ):
  """Low-level method to sync quadrant data with the contiguous arrays.
  """

  cdef:
    p8est_tree_t* tree = NULL
    p8est_quadrant_t* quads = NULL
    p8est_quadrant_t* cell = NULL
    aux_quadrant_data_t* cell_aux

    npy.npy_int32 i = 0
    npy.npy_int32 j = 0
    npy.npy_int32 k = 0
    npy.npy_int32 m = 0
    npy.npy_int32 q = 0

    npy.npy_int32 root_idx = 0
    npy.npy_int32 adapt_idx = 0
    npy.npy_int8 max_prev_adapt = 0

    npy.npy_int32 cell_idx = 0

    npy.npy_int32 prev_cell_idx = 0
    npy.npy_int32[:,:,:] prev_fine_idx

    npy.npy_int32 cell_adj_idx = 0
    npy.npy_int8 cell_adj_face_idx = 0
    npy.npy_int8 face_order = 0

  for root_idx in range(first_local_tree, last_local_tree+1):
    tree = &trees[root_idx]
    quads = <p8est_quadrant_t*>tree.quadrants.array

    for q in range(tree.quadrants.elem_count):
      cell = &quads[q]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data

      cell_idx = tree.quadrants_offset + q
      # print(f"idx= {cell_aux.idx}, adapted= {cell_aux.adapted}, adapt_idx= {cell_aux.adapt_idx}")

      # finish tracking of changes due to refine/coarsen operation
      # now that the indices are known, and thus able to map previous quadrants
      if cell_aux.adapted == 1:
        # refined
        leaf_adapted[adapt_idx] = 1

        #  which octant of the replaced cell
        m = (
          (cell_aux.replaced_idx[1] >= 0)
          + 2*(cell_aux.replaced_idx[2] >= 0)
          + 3*(cell_aux.replaced_idx[3] >= 0)
          + 4*(cell_aux.replaced_idx[4] >= 0)
          + 5*(cell_aux.replaced_idx[5] >= 0)
          + 6*(cell_aux.replaced_idx[6] >= 0)
          + 7*(cell_aux.replaced_idx[7] >= 0) )

        if m == 0:
          prev_cell_idx = cell_aux.replaced_idx[0]
          leaf_adapted_coarse[adapt_idx] = prev_cell_idx

        k = m // 4
        m %= 4
        j = m // 2
        i = m % 2

        leaf_adapted_fine[adapt_idx, k, j, i] = cell_idx
        adapt_idx += 1

      elif cell_aux.adapted == -1:
        # coarsened
        leaf_adapted[adapt_idx] = -1
        leaf_adapted_coarse[adapt_idx] = cell_idx

        prev_fine_idx = leaf_adapted_fine[adapt_idx]
        prev_fine_idx[0,0,0] = cell_aux.replaced_idx[0]
        prev_fine_idx[0,0,1] = cell_aux.replaced_idx[1]
        prev_fine_idx[0,1,0] = cell_aux.replaced_idx[2]
        prev_fine_idx[0,1,1] = cell_aux.replaced_idx[3]
        prev_fine_idx[1,0,0] = cell_aux.replaced_idx[4]
        prev_fine_idx[1,0,1] = cell_aux.replaced_idx[5]
        prev_fine_idx[1,1,0] = cell_aux.replaced_idx[6]
        prev_fine_idx[1,1,1] = cell_aux.replaced_idx[7]

        adapt_idx += 1

      else:
        #??
        pass

      # reset auxiliary information
      cell_aux.idx = cell_idx
      cell_aux.adapted = 0

      adapt[cell_idx] = cell_aux.adapt
      weight[cell_idx] = cell_aux.weight

      root[cell_idx] = root_idx
      level[cell_idx] = cell.level
      origin[cell_idx,0] = cell.x
      origin[cell_idx,1] = cell.y
      origin[cell_idx,2] = cell.z

      # get adjacency information from the mesh
      # loop x, y, z
      for i in range(3):
        # loop -x,+x, .
        for j in  range(2):
          cell_adj_idx = quad_to_quad[cell_idx,i,j]
          cell_adj_face_idx = quad_to_face[cell_idx,i,j]

          # A value of v = 0..23 indicates one same-size neighbor.
          # v = r * 6 + nf
          # nf = 0..5
          # r = 0..3 is the relative orientation
          # A value of v = 24..119 indicates a double-size neighbor.
          # designates the subface of the large neighbor that the quadrant
          # touches (this is the same as the large neighbor's face corner).
          # v = 24 + h * 24 + r * 6 + nf
          # h = 0..3 is the number of the subface.
          # A value of v = -24..-1 indicates four half-size neighbors.
          face_order = cell_adj_face_idx % 24

          cell_adj_face[cell_idx,i,j] = face_order % 6
          cell_adj_order[cell_idx,i,j] = face_order // 6

          # v -> 0, or 1 + h
          m = cell_adj_face_idx // 24

          cell_adj_subface[cell_idx,i,j] = 0 if m == 0 else (m - 1)


          if cell_adj_face_idx >= 0:
            # neighbor <= level
            cell_adj[cell_idx,i,j,:,:] = cell_adj_idx
            cell_adj_level[cell_idx,i,j] = 0 if (cell_adj_face_idx < 8) else -1
          else:
            # neighbor > level
            # In this case the corresponding quad_to_quad index points into the
            # quad_to_half array that stores four quadrant numbers per index,
            cell_adj[cell_idx,i,j,:,:] = quad_to_half[cell_adj_idx]
            cell_adj_level[cell_idx,i,j] = 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef _sync_ghost_info(
  p8est_quadrant_t* ghosts,
  npy.npy_int32 num_ghosts,
  const npy.npy_int32[:] proc_offsets,
  # out
  npy.npy_int32[:] rank,
  npy.npy_int32[:] root,
  npy.npy_int32[:] idx,
  npy.npy_int8[:] level,
  npy.npy_int32[:,:] origin ):


  cdef:
    p8est_quadrant_t* cell = NULL
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
      origin[q,2] = cell.z

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# @cython.boundscheck(False)
# @cython.wraparound(False)
cdef _sync_mirror_idx(
  p8est_quadrant_t* mirrors,
  npy.npy_int32 num_mirrors,
  # out
  npy.npy_int32[:] mirrors_idx ):


  cdef:
    p8est_quadrant_t* cell = NULL
    npy.npy_int32 q = 0

  for q in range(num_mirrors):
    cell = &mirrors[q]
    mirrors_idx[q] = cell.p.piggy3.local_num

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _init_quadrant(
  p8est_t* p4est,
  p4est_topidx_t root_idx,
  p8est_quadrant_t* quadrant ) nogil:
  # print(f"+ quadrant: root= {root_idx}, level= {quadrant.level}, x= {quadrant.x}, y=, {quadrant.y}, data= {<int>quadrant.p.user_data})")
  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data
  cell_aux.idx = -1
  cell_aux.adapt = 0
  cell_aux.weight = 1

  cell_aux.adapted = 0
  cell_aux.replaced_idx[0] = -1
  cell_aux.replaced_idx[1] = -1
  cell_aux.replaced_idx[2] = -1
  cell_aux.replaced_idx[3] = -1
  cell_aux.replaced_idx[4] = -1
  cell_aux.replaced_idx[5] = -1
  cell_aux.replaced_idx[6] = -1
  cell_aux.replaced_idx[7] = -1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _replace_quadrants(
  p8est_t* p4est,
  p4est_topidx_t root_idx,
  int num_outgoing,
  p8est_quadrant_t* outgoing[],
  int num_incoming,
  p8est_quadrant_t* incoming[] ) nogil:

  cdef:
    p8est_quadrant_t* cell
    aux_quadrant_data_t* cell_aux

    p8est_quadrant_t* _cell
    aux_quadrant_data_t* _cell_aux

    npy.npy_int8 prev_adapt
    npy.npy_int8 prev_weight

    npy.npy_int8 k

  # NOTE: incoming means the 'added' quadrants, and outgoing means 'removed'
  if num_outgoing == 8:
    # Coarsening: remove 8 -> add 1
    # assert num_outgoing == 8
    # assert num_incoming == 1

    # print(f"Coarsening: root= {root_idx}")

    cell = incoming[0]

    # flag that this index currently refers to the adapt array
    cell_aux = <aux_quadrant_data_t*>cell.p.user_data
    cell_aux.adapted = -1

    prev_adapt = -128
    prev_weight = 0

    for k in range(8):
      _cell = outgoing[k]
      _cell_aux = <aux_quadrant_data_t*>_cell.p.user_data

      prev_adapt = max(prev_adapt, _cell_aux.adapt)
      prev_weight = max(prev_weight, _cell_aux.weight)

      cell_aux.replaced_idx[k] = _cell_aux.idx

    cell_aux.adapt = prev_adapt + 1
    cell_aux.weight = prev_weight

  else:
    # Refining: remove 1 -> add 8
    # assert num_outgoing == 1
    # assert num_incoming == 8
    # print(f"Refining: root= {root_idx}")

    _cell = outgoing[0]
    _cell_aux = <aux_quadrant_data_t*>_cell.p.user_data

    prev_adapt = _cell_aux.adapt
    prev_weight = _cell_aux.weight

    for k in range(8):
      cell = incoming[k]

      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.adapt = prev_adapt - 1
      cell_aux.weight = prev_weight

      cell_aux.adapted = 1
      cell_aux.replaced_idx[k] = _cell_aux.idx
      cell_aux.replaced_idx[(k+1)%8] = -1
      cell_aux.replaced_idx[(k+2)%8] = -1
      cell_aux.replaced_idx[(k+3)%8] = -1
      cell_aux.replaced_idx[(k+4)%8] = -1
      cell_aux.replaced_idx[(k+5)%8] = -1
      cell_aux.replaced_idx[(k+6)%8] = -1
      cell_aux.replaced_idx[(k+7)%8] = -1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _refine_quadrant(
  p8est_t* p4est,
  p4est_topidx_t root_idx,
  p8est_quadrant_t* quadrant ) nogil:

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return cell_aux.adapt > 0

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _coarsen_quadrants(
  p8est_t* p4est,
  p4est_topidx_t root_idx,
  p8est_quadrant_t* quadrants[] ) nogil:

  cdef aux_quadrant_data_t* cell_aux

  for k in range(8):
    cell_aux = <aux_quadrant_data_t*>quadrants[k].p.user_data

    if cell_aux.adapt >= 0:
      return 0

  return 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _weight_quadrant(
  p8est_t* p4est,
  p4est_topidx_t root_idx,
  p8est_quadrant_t* quadrant ) nogil:

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return cell_aux.weight

