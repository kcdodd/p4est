


from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )

cimport numpy as cnp
import numpy as np

from mpi4py import MPI
from mpi4py.MPI cimport MPI_Comm, Comm

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from p4est.mesh.quad_mesh import (
  QuadMeshBase,
  QuadMesh )

cdef extern from "Python.h":
  const int PyBUF_READ
  const int PyBUF_WRITE
  PyObject *PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
QUAD_ADAPTED_FINE = 0x10
QUAD_ADAPTED_COARSE = 0x01

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class LeafInfo:
  _spec = {
    # the index of the original root level mesh.cells
    'root' : (tuple(), np.int32),
    # refinement level
    'level' : (tuple(), np.int8),
    # Normalized coordinate of the leaf's origin relative to the root cell
    # stored as integer units to allow exact arithmetic:
    # 0 -> 0.0
    # 2**max_level -> 1.0
    # To get positions from this origin, the relative width of the leaf can be
    # computed from the refinement level:
    # 2**(max_level - level) -> 1.0/2**level
    # NOTE: This results in higher precision than a normalized 32bit float,
    # since a single-precision float only has 24bits for the fraction.
    # Floating point arithmetic involving the normalized coordinates should
    # use 64bit (double) precision to avoid loss of precision.
    'origin' : ((2,), np.int32),
    # Computational weight of the leaf used for load partitioning among processors
    'weight' : (tuple(), np.int32),
    # A flag used to control refinement (>0) and coarsening (<0)
    'adapt' : (tuple(), np.int8),
    # Indices of up to 6 unique adjacent cells, ordered as:
    #    |110|111|
    # ---+---+---+---
    # 001|       |011
    # ---+       +---
    # 000|       |010
    # ---+---+---+---
    #    |100|101|
    #
    # Indexing: [(y-normal, x-normal), (-face, +face), (-half, +half)]
    # If cell_adj[i,j,0] == cell_adj[i,j,1], then both halfs are shared with
    # the same cell (of equal or greater size).
    # Otherwise each half is shared with different cells of 1/2 size.
    'cell_adj' : ((2,2,2), np.int32),
    # If the neighbor is larger, then the value of cell_adj_face may be in the
    # range 0..15 (instead of 0..7), where the additional bit indicates the
    # sub-face. E.G.
    # sub_face, face_orientation = divmod(cell_adj_face, 8)
    # orientation, face = divmod(face_orientation, 4)
    'cell_adj_face' : ((2,2), np.int8) }

  #-----------------------------------------------------------------------------
  def __init__(self, *args, **kwargs):
    if len(args) and len(kwargs):
      raise ValueError(f"Arguments must be only size, or all info values")

    if len(args):
      if len(args) == 1:
        if isinstance(args[0], int):
          self.resize(args[0])
        else:
          self.set_from(args[0])
      else:
        self.set_from(args)
    elif len(kwargs):
      self.set_from(kwargs)

    else:
      self.resize(0)

  #-----------------------------------------------------------------------------
  def as_tuple(self):
    return LeafInfoTuple(
      root = self._root,
      level = self._level,
      origin = self._origin,
      weight = self._weight,
      adapt = self._adapt,
      cell_adj = self._cell_adj,
      cell_adj_face = self._cell_adj_face )

  #-----------------------------------------------------------------------------
  def set_from(self, info):
    if not isinstance(info, LeafInfoTuple):
      if isinstance(info, Mapping):
        info = LeafInfoTuple(**info)
      else:
        info = LeafInfoTuple(*info)

    size = None
    for (k, (shape, dtype)), v in zip(self._spec.items(), info):
      if v.shape[1:] != shape or v.dtype != dtype:
        raise ValueError(f"'{k}' must have shape[1:] == {shape}, dtype == {dtype}: {v.shape[1:]}, {v.dtype}")

      if size is None:
        size = len(v)

      elif len(v) != size:
        raise ValueError(f"All arrays must have same length: {k} -> {len(v)}")

    self._root = np.ascontiguousarray(info.root)
    self._level = np.ascontiguousarray(info.level)
    self._origin = np.ascontiguousarray(info.origin)
    self._weight = np.ascontiguousarray(info.weight)
    self._adapt = np.ascontiguousarray(info.adapt)
    self._cell_adj = np.ascontiguousarray(info.cell_adj)
    self._cell_adj_face = np.ascontiguousarray(info.cell_adj_face)

  #-----------------------------------------------------------------------------
  @property
  def root(self):
    return self._root

  #-----------------------------------------------------------------------------
  @root.setter
  def root(self, val):
    self._root[:] = val

  #-----------------------------------------------------------------------------
  @property
  def level(self):
    return self._level

  #-----------------------------------------------------------------------------
  @level.setter
  def level(self, val):
    self._level[:] = val

  #-----------------------------------------------------------------------------
  @property
  def origin(self):
    return self._origin

  #-----------------------------------------------------------------------------
  @origin.setter
  def origin(self, val):
    self._origin[:] = val

  #-----------------------------------------------------------------------------
  @property
  def weight(self):
    return self._weight

  #-----------------------------------------------------------------------------
  @weight.setter
  def weight(self, val):
    self._weight[:] = val

  #-----------------------------------------------------------------------------
  @property
  def adapt(self):
    return self._adapt

  #-----------------------------------------------------------------------------
  @adapt.setter
  def adapt(self, val):
    self._adapt[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self):
    return self. _cell_adj

  #-----------------------------------------------------------------------------
  @cell_adj.setter
  def cell_adj(self, val):
    self._cell_adj[:] = val

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self):
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @cell_adj_face.setter
  def cell_adj_face(self, val):
    self._cell_adj_face[:] = val

  #-----------------------------------------------------------------------------
  def resize(self, size):
    if self._root is not None and size == len(self._root):
      return

    info = list()

    for (k, (shape, dtype)), _arr in zip(self._spec.items(), self.as_tuple()):

      arr = np.zeros((size, *shape), dtype = dtype)

      if _arr is not None:
        arr[:len(_arr)] = _arr[:size]

      info.append(arr)

    self.set_from(info)

  #-----------------------------------------------------------------------------
  def __len__(self):
    return len(self._root)

  #-----------------------------------------------------------------------------
  def __getitem__( self, idx ):
    return LeafInfo(
      np.ascontiguousarray(arr[idx])
      for arr in self.as_tuple() )

  #-----------------------------------------------------------------------------
  def __setitem__( self, idx, info ):
    if not isinstance(info, (LeafInfo, LeafInfoTuple)):
      raise ValueError(f"Expected LeafInfo: {type(info)}")

    if isinstance(info, LeafInfo):
      info = info.as_tuple()

    for a, b in zip(self.as_tuple(), info):
      a[idx] = b

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LeafInfoTuple = namedtuple('LeafInfoTuple', list(LeafInfo._spec.keys()))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:
  """

  Parameters
  ----------
  mesh : QuadMesh
  min_quadrants : None | int
    (default: 0)
  min_level : None | int
    (default: 0)
  fill_uniform : bool
  comm : None | mpi4py.MPI.Comm
    (default: mpi4py.MPI.COMM_WORLD)

  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    mesh,
    min_quadrants = None,
    min_level = None,
    fill_uniform = None,
    comm = None ):

    #...........................................................................
    if not isinstance(mesh, QuadMeshBase):
      raise ValueError(f"mesh must be a QuadMeshBase: {type(mesh)}")

    if min_quadrants is None:
      min_quadrants = 0

    if min_level is None:
      min_level = 0

    if fill_uniform is None:
      fill_uniform = False

    if comm is None:
      comm = MPI.COMM_WORLD


    #...........................................................................
    self._max_level = P4EST_MAXLEVEL
    self._comm = comm
    self._mesh = mesh
    self._leaf_info = LeafInfo(0)

    self._leaf_adapt_idx = 0
    self._leaf_adapt_coarse = None
    self._leaf_adapt_fine = None


    self._init(min_quadrants, min_level, fill_uniform)

  #-----------------------------------------------------------------------------
  cdef _init(
    P4est self,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform ):

    memset(&self._connectivity, 0, sizeof(p4est_connectivity_t));

    cdef np.ndarray[double, ndim=2] verts = self._mesh.verts
    cdef np.ndarray[np.npy_int32, ndim=3] cells = self._mesh.cells
    cdef np.ndarray[np.npy_int32, ndim=3] cell_adj = self._mesh.cell_adj
    cdef np.ndarray[np.npy_int8, ndim=3] cell_adj_face = self._mesh.cell_adj_face

    cdef np.ndarray[np.npy_int32, ndim=1] tree_to_corner = self._mesh.cell_nodes.ravel()
    cdef np.ndarray[np.npy_int32, ndim=1] ctt_offset = self._mesh.node_cells.row_idx
    cdef np.ndarray[np.npy_int32, ndim=1] corner_to_tree = self._mesh.node_cells.data
    cdef np.ndarray[np.npy_int8, ndim=1] corner_to_corner = self._mesh.node_cells_vert.data

    self._connectivity.num_vertices = len(verts)
    self._connectivity.vertices = <double*>verts.data

    self._connectivity.num_trees = len(cells)
    self._connectivity.tree_to_vertex = <np.npy_int32*>(cells.data)
    self._connectivity.tree_to_tree = <np.npy_int32*>(cell_adj.data)
    self._connectivity.tree_to_face = <np.npy_int8*>(cell_adj_face.data)

    self._connectivity.num_corners = len(ctt_offset)-1
    self._connectivity.ctt_offset = <np.npy_int32*>(ctt_offset.data)

    if self._connectivity.num_corners > 0:
      # NOTE: intially NULL from memset
      self._connectivity.tree_to_corner = <np.npy_int32*>(tree_to_corner.data)
      self._connectivity.corner_to_tree = <np.npy_int32*>(corner_to_tree.data)
      self._connectivity.corner_to_corner = <np.npy_int8*>(corner_to_corner.data)


    cdef p4est_t* p4est = NULL
    cdef sc_MPI_Comm comm = <sc_MPI_Comm> (<Comm>self.comm).ob_mpi
    cdef p4est_connectivity_t* connectivity = &(self._connectivity)

    with nogil:
      p4est = p4est_new_ext(
        comm,
        connectivity,
        min_quadrants,
        min_level,
        fill_uniform,
        sizeof(aux_quadrant_data_t),
        <p4est_init_t>_init_quadrant,
        <void*>self )

    self._p4est = p4est
    self._update_iter()

  #-----------------------------------------------------------------------------
  # @cython.boundscheck(False)
  # @cython.wraparound(False)
  cdef _update_iter(P4est self):
    cdef p4est_ghost_t* ghost = NULL
    cdef p4est_mesh_t* mesh = NULL


    cdef p4est_tree_t* trees = <p4est_tree_t*>self._p4est.trees.array
    cdef p4est_tree_t* tree = NULL

    cdef p4est_quadrant_t* quads = NULL
    cdef p4est_quadrant_t* cell = NULL
    cdef aux_quadrant_data_t* cell_aux

    cdef cnp.npy_intp cell_idx = 0
    cdef cnp.npy_intp face_idx = 0
    cdef cnp.npy_intp cell_adj_idx = 0
    cdef cnp.npy_intp cell_adj_face_idx = 0

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

    cdef np.ndarray[cnp.npy_int32, ndim = 3] quad_to_quad = ndarray_from_ptr(
      write = False,
      dtype = np.int32,
      count = 4*mesh.local_num_quadrants,
      arr = <char*>mesh.quad_to_quad).reshape(-1,2,2)

    cdef np.ndarray[cnp.npy_int8, ndim = 3] quad_to_face = ndarray_from_ptr(
      write = False,
      dtype = np.int8,
      count = 4*mesh.local_num_quadrants,
      arr = <char*>mesh.quad_to_face).reshape(-1,2,2)

    cdef np.ndarray[cnp.npy_int32, ndim = 2] quad_to_half = ndarray_from_sc_array(
      write = False,
      dtype = np.int32,
      subitems = 2,
      arr = mesh.quad_to_half).reshape(-1,2)

    prev_leaf_info = self._leaf_info
    cdef np.ndarray[cnp.npy_int8, ndim = 1] prev_adapt = prev_leaf_info.adapt

    self._leaf_info = LeafInfo(mesh.local_num_quadrants)
    cdef np.ndarray[cnp.npy_int32, ndim = 1] root = self._leaf_info.root
    cdef np.ndarray[cnp.npy_int8, ndim = 1] level = self._leaf_info.level
    cdef np.ndarray[cnp.npy_int32, ndim = 2] origin = self._leaf_info.origin
    cdef np.ndarray[cnp.npy_int32, ndim = 1] weight = self._leaf_info.weight
    cdef np.ndarray[cnp.npy_int8, ndim = 1] adapt = self._leaf_info.adapt
    cdef np.ndarray[cnp.npy_int32, ndim = 4] cell_adj = self._leaf_info.cell_adj
    cdef np.ndarray[cnp.npy_int8, ndim = 3] cell_adj_face = self._leaf_info.cell_adj_face

    from_mask = np.ones(len(prev_leaf_info), dtype = bool)
    to_mask = np.ones(len(self._leaf_info), dtype = bool)

    for root_idx in range(self._p4est.first_local_tree, self._p4est.last_local_tree+1):
      tree = &trees[root_idx]
      quads = <p4est_quadrant_t*>tree.quadrants.array

      for q in range(tree.quadrants.elem_count):
        cell = &quads[q]
        cell_aux = <aux_quadrant_data_t*>cell.p.user_data

        cell_idx = tree.quadrants_offset + q
        # print(f"idx= {cell_aux.idx}, adapted= {cell_aux.adapted}, adapt_idx= {cell_aux.adapt_idx}")

        # finish tracking of changes due to refine/coarsen operation
        # now that the indices are known, and thus able to map previous quadrants
        if cell_aux.adapted == 1:
          adapt_idx, k = divmod(cell_aux.adapt_idx, 4)
          i, j = divmod(k, 2)
          self._leaf_adapt_fine[adapt_idx, i, j] = cell_idx

          prev_cell_idx = self._leaf_adapt_coarse[adapt_idx]
          adapt[cell_idx] = prev_adapt[prev_cell_idx] - 1

          from_mask[prev_cell_idx] = False
          to_mask[cell_idx] = False

        elif cell_aux.adapted == -1:
          self._leaf_adapt_coarse[cell_aux.adapt_idx] = cell_idx

          prev_cell_idx = self._leaf_adapt_fine[cell_aux.adapt_idx]
          adapt[cell_idx] = 1 + np.amax(prev_adapt[prev_cell_idx])

          from_mask[prev_cell_idx] = False
          to_mask[cell_idx] = False

        else:
          #??
          pass

        # reset auxiliary information
        cell_aux.idx = cell_idx
        cell_aux.adapt_idx = -1
        cell_aux.adapted = 0

        root[cell_idx] = root_idx
        level[cell_idx] = cell.level
        origin[cell_idx] = (cell.x, cell.y)

        # get adjacency information from the mesh
        for i in range(2):
          for j in  range(2):
            cell_adj_idx = quad_to_quad[cell_idx,i,j]
            cell_adj_face_idx = quad_to_face[cell_idx,i,j]

            # A value of v = 0..7 indicates one same-size neighbor.
            # A value of v = 8..23 indicates a double-size neighbor.
            #   8 % 8 -> face 0
            #   8 // 16 -> subface 0
            #   15 % 8 -> face 7
            #   15 // 16 -> subface 0
            # designates the subface of the large neighbor that the quadrant
            # touches (this is the same as the large neighbor's face corner).
            #   16 % 8 -> face 0
            #   16 // 16 -> subface 1
            #   23 % 8 -> face 7
            #   23 // 16 -> subface 1
            # A value of v = -8..-1 indicates two half-size neighbors.
            #   -8 % 8 -> 0
            #   -1 % 8 -> 7
            # TODO: store face orientation separatly (r * 4 + nf)
            # TODO: store double-size neighbor subface somewhere
            cell_adj_face[cell_idx,i,j] = cell_adj_face_idx % 8

            if cell_adj_face_idx >= 0:
              cell_adj[cell_idx,i,j] = cell_adj_idx
            else:
              # In this case the corresponding quad_to_quad index points into the
              # quad_to_half array that stores two quadrant numbers per index,
              cell_adj[cell_idx,i,j] = quad_to_half[cell_adj_idx]


    if len(prev_leaf_info) > 0:
      # copy data that should not change
      adapt[to_mask] = prev_adapt[from_mask]

    p4est_mesh_destroy(mesh)

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

    self._leaf_info = None
    self._leaf_adapt_coarse = None
    self._leaf_adapt_fine = None
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
      uv = np.zeros((2,), dtype = np.float64)

    else:
      uv = np.asarray(uv, dtype = np.float64)

    if idx is None:
      idx = slice(None)

    info = self._leaf_info[idx]

    # compute the local discrete width of the leaf within the root cell
    # NOTE: the root level = 0 is 2**P4EST_MAXLEVEL wide, and all refinement
    # levels are smaller by factors that are powers of 2.
    qwidth = np.left_shift(1, P4EST_MAXLEVEL - info.level.astype(np.int32))

    cell_verts = self.mesh.verts[ self.mesh.cells[ info.root ] ]

    # compute the relative position within the root cell
    UV = np.clip(
      ( info.origin + uv[None,:] * qwidth[:,None] ) / P4EST_ROOT_LEN,
      0.0, 1.0)

    # Compute the coefficients for bilinear interpolation of the root cells
    # vertices absolute position onto the desired position relative to the leaf.

    if interp is None:
      _UV = 1.0 - UV

      # bi-linear interpolation
      c = np.empty(cell_verts.shape[:3])
      c[:,0,0] = _UV[:,0]*_UV[:,1]
      c[:,0,1] = UV[:,0]*_UV[:,1]
      c[:,1,0] = _UV[:,0]*UV[:,1]
      c[:,1,1] = UV[:,0]*UV[:,1]

      # Perform the interpolation
      return np.sum(c[:,:,:,None] * cell_verts, axis = (1,2))

    else:
      return interp(cell_verts, UV)

  #-----------------------------------------------------------------------------
  def refine(self,
    recursive = False,
    maxlevel = -1 ):

    num_refine = np.count_nonzero(self._leaf_info.adapt > 0)

    leaf_adapt_coarse = -np.ones(
      (num_refine,),
      dtype = np.intp)

    leaf_adapt_fine = -np.ones(
      (num_refine, 2, 2),
      dtype = np.intp )

    self._leaf_adapt_idx = 0
    self._leaf_adapt_coarse = leaf_adapt_coarse
    self._leaf_adapt_fine = leaf_adapt_fine

    self._refine(int(recursive), int(maxlevel))

    # save the information of the quads being refined before it's lost
    coarse_info = self._leaf_info[leaf_adapt_coarse]

    self._update_iter()

    self._leaf_adapt_idx = 0
    self._leaf_adapt_coarse = None
    self._leaf_adapt_fine = None


    return coarse_info, leaf_adapt_coarse, leaf_adapt_fine

  #-----------------------------------------------------------------------------
  cdef _refine(
    P4est self,
    int recursive,
    int maxlevel ):

    with nogil:
      p4est_refine_ext(
        self._p4est,
        recursive,
        maxlevel,
        <p4est_refine_t>_refine_quadrant,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

  #-----------------------------------------------------------------------------
  def coarsen(self,
    recursive = False,
    orphans = False ):

    # NOTE: this should over-estimate the number of coarsened cells, since
    # the condition for coarsening is more strict
    num_coarse = np.count_nonzero(self._leaf_info.adapt < 0) // 4

    leaf_adapt_coarse = -np.ones(
      (num_coarse,),
      dtype = np.intp)

    leaf_adapt_fine = -np.ones(
      (num_coarse, 2, 2),
      dtype = np.intp )

    self._leaf_adapt_idx = 0
    self._leaf_adapt_coarse = leaf_adapt_coarse
    self._leaf_adapt_fine = leaf_adapt_fine

    self._coarsen(int(recursive), int(orphans))

    # save the information of the quads being coarsened before it's lost
    fine_info = self._leaf_info[leaf_adapt_fine]

    self._update_iter()

    # trim output arrays to actual length
    fine_info = fine_info[:self._leaf_adapt_idx]
    leaf_adapt_fine = leaf_adapt_fine[:self._leaf_adapt_idx]
    leaf_adapt_coarse = leaf_adapt_coarse[:self._leaf_adapt_idx]

    self._leaf_adapt_idx = 0
    self._leaf_adapt_coarse = None
    self._leaf_adapt_fine = None

    return fine_info, leaf_adapt_fine, leaf_adapt_coarse

  #-----------------------------------------------------------------------------
  cdef _coarsen(
    P4est self,
    int recursive,
    int orphans ):

    with nogil:
      p4est_coarsen_ext(
        self._p4est,
        recursive,
        orphans,
        <p4est_coarsen_t>_coarsen_quadrants,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

  #-----------------------------------------------------------------------------
  def balance(self,
    connections = None ):

    if connections is None:
      connections = 'full'

    connections = {
      'face': P4EST_CONNECT_FACE,
      'corner': P4EST_CONNECT_CORNER,
      'full': P4EST_CONNECT_FULL }[connections]

    self._balance(<p4est_connect_type_t>connections)

  #-----------------------------------------------------------------------------
  cdef _balance(
    P4est self,
    p4est_connect_type_t btype ):

    with nogil:
      p4est_balance_ext(
        self._p4est,
        btype,
        <p4est_init_t>_init_quadrant,
        <p4est_replace_t> _replace_quadrants )

  #-----------------------------------------------------------------------------
  def partition(self,
    allow_coarsening = False):

    self._partition(int(allow_coarsening))

  #-----------------------------------------------------------------------------
  cdef _partition(
    P4est self,
    int allow_coarsening ):

    with nogil:
      p4est_partition_ext(
        self._p4est,
        allow_coarsening,
        <p4est_weight_t>_weight_quadrant )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _init_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) with gil:
  # print(f"+ quadrant: root= {root_idx}, level= {quadrant.level}, x= {quadrant.x}, y=, {quadrant.y}, data= {<int>quadrant.p.user_data})")
  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data
  cell_aux.idx = -1
  cell_aux.adapt_idx = -1
  cell_aux.adapted = 0

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _replace_quadrants(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  int num_outgoing,
  p4est_quadrant_t* outgoing[],
  int num_incoming,
  p4est_quadrant_t* incoming[] ) with gil:

  self = <P4est>p4est.user_pointer
  cdef p4est_quadrant_t* cell
  cdef aux_quadrant_data_t* cell_aux

  cdef size_t adapt_idx = self._leaf_adapt_idx
  self._leaf_adapt_idx += 1

  assert adapt_idx < len(self._leaf_adapt_coarse)

  # NOTE: incoming means the 'added' quadrants, and outgoing means 'removed'
  if num_outgoing == 4:
    # Coarsening: remove 4 -> add 1
    # assert num_outgoing == 4
    assert num_incoming == 1

    # print(f"Coarsening: root= {root_idx}, adapt_idx= {adapt_idx}")

    cell = incoming[0]

    # flag that this index currently refers to the adapt array
    cell_aux = <aux_quadrant_data_t*>cell.p.user_data
    cell_aux.adapted = -1
    cell_aux.adapt_idx = adapt_idx

    for k in range(4):
      cell = outgoing[k]
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data

      i, j = divmod(k, 2)
      self._leaf_adapt_fine[adapt_idx, i, j] = cell_aux.idx
      # print(f" - [{i,j}]: cell_idx= {cell_aux.idx}")

  else:
    # Refining: remove 1 -> add 4
    assert num_outgoing == 1
    assert num_incoming == 4

    # print(f"Refining: root= {root_idx}, adapt_idx= {adapt_idx}")

    cell = outgoing[0]
    cell_aux = <aux_quadrant_data_t*>cell.p.user_data

    # store index of cell being refined
    self._leaf_adapt_coarse[adapt_idx] = cell_aux.idx

    # print(f" - : cell_idx= {cell_aux.idx}")

    for k in range(4):
      cell = incoming[k]

      # flag that this index currently refers to the adapt array
      cell_aux = <aux_quadrant_data_t*>cell.p.user_data
      cell_aux.adapted = 1
      cell_aux.adapt_idx = 4*adapt_idx + k

      # i, j = divmod(k, 2)
      # print(f" + [{i},{j}]: adapt_idx= {cell_aux.adapt_idx}")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: these are not class members because they need to match the callback sig.
# but are implemented as though they were by casting the user pointer to be 'self'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _refine_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) with gil:

  self = <P4est>p4est.user_pointer

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return self._leaf_info._adapt[cell_aux.idx] > 0

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _coarsen_quadrants(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrants[] ) with gil:

  self = <P4est>p4est.user_pointer
  cdef aux_quadrant_data_t* cell_aux

  for k in range(4):
    cell_aux = <aux_quadrant_data_t*>quadrants[k].p.user_data

    if self._leaf_info._adapt[cell_aux.idx] >= 0:
      return 0

  return 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _weight_quadrant(
  p4est_t* p4est,
  p4est_topidx_t root_idx,
  p4est_quadrant_t* quadrant ) with gil:

  self = <P4est>p4est.user_pointer

  cdef aux_quadrant_data_t* cell_aux = <aux_quadrant_data_t*>quadrant.p.user_data

  return self._leaf_info._weight[cell_aux.idx]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _iter_volume(
  p4est_iter_volume_info_t* info,
  void *user_data ) with gil:
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef ndarray_from_ptr(write, dtype, count, char* arr):
  itemsize = np.dtype(dtype).itemsize

  buffer = <object>PyMemoryView_FromMemory(
    arr,
    itemsize * count,
    PyBUF_WRITE if write else PyBUF_READ )

  return np.frombuffer(
    buffer,
    dtype = dtype,
    count = count )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef ndarray_from_sc_array(write, dtype, subitems, sc_array_t* arr):
  itemsize = np.dtype(dtype).itemsize

  if itemsize * subitems != arr.elem_size:
    raise ValueError(f"Item size does not match: {dtype}.itemsize == {itemsize}, arr.elem_size == {arr.elem_size}")

  buffer = <object>PyMemoryView_FromMemory(
    arr.array,
    arr.elem_size * arr.elem_count,
    PyBUF_WRITE if write else PyBUF_READ )

  return np.frombuffer(
    buffer,
    dtype = dtype,
    count = subitems * arr.elem_count )
