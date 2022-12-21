



cimport numpy as cnp
import numpy as np

from mpi4py import MPI
from mpi4py.MPI cimport MPI_Comm, Comm

from libc.stdlib cimport malloc, free
from libc.string cimport memset

from p4est.mesh.quad_mesh import QuadMesh

cdef extern from "Python.h":
  const int PyBUF_READ
  const int PyBUF_WRITE
  PyObject *PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:
  """

  Parameters
  ----------
  mesh : QuadMesh

  shape : None | tuple[int]
    The shape of the data stored in each leaf (default: tuple()).

    .. note::

      The default shape = tuple() is interpreted as a single scalar value per leaf,
      storing the same amount of data as shapes of (1,) or (1,1) etc.

  dtype : None | np.dtype
    The type of the data stored in each leaf quadrant. No data is allocated
    unless this is specified.

    .. note::

      The top-level dtype may be a composition of other dtypes defined for
      NumPy structured array (https:#numpy.org/doc/stable/user/basics.rec.html).
      This gives some freedom to define either an array of structs (by setting
      the above shape), a struct of arrays (by setting a shape in the dtype's
      fields), or a combination.

  cell_adj : None | np.ndarray with shape (NC, 4) and dtype np.int32
    If not given, the adjacency is computed from the cells array.
  cell_adj_face : None | np.ndarray with shape (NC, 4) and dtype np.int8
    If not given, the adjacency is computed from the cells array.
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
    shape = None,
    dtype = None,
    min_quadrants = None,
    min_level = None,
    fill_uniform = None,
    comm = None ):

    #...........................................................................
    if not isinstance(mesh, QuadMesh):
      raise ValueError(f"mesh must be a QuadMesh: {type(mesh)}")

    if min_quadrants is None:
      min_quadrants = 0

    if min_level is None:
      min_level = 0

    if fill_uniform is None:
      fill_uniform = False

    if comm is None:
      comm = MPI.COMM_WORLD

    #...........................................................................
    # define quadrant data

    if shape is None:
      shape = tuple()

    shape = tuple(int(s) for s in shape)

    if dtype is not None:
      dtype = np.dtype(dtype)

    #...........................................................................
    self._comm = comm
    self._mesh = mesh
    self._shape = shape
    self._dtype = dtype

    self._leaf_dtype = np.dtype([
      ('cell', np.int32),
      ('lvl', np.int8),
      ('ij', np.int32, (2,)),
      ('weight', np.int32),
      ('refine', np.int8) ])

    self._leaf_info = np.zeros(
      self._mesh.cells.shape[:1],
      dtype = self._leaf_dtype )

    self._leaf_info['cell'] = -1
    self._leaf_count = 0

    self._init(min_quadrants, min_level, fill_uniform)

    self._leaf_info = self._leaf_info[:self._leaf_count]

    if self._dtype is not None:
      self._leaf_data = np.zeros(
        (self._leaf_count, *self._shape),
        dtype = self._dtype)

    else:
      self._leaf_data = None

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
    cdef np.ndarray[np.npy_int32, ndim=1] corner_to_cell_offset = self._mesh.corner_to_cell_offset

    self._connectivity.num_vertices = len(verts)
    self._connectivity.vertices = <double*>verts.data

    self._connectivity.num_trees = len(cells)
    self._connectivity.tree_to_vertex = <np.npy_int32*>(cells.data)
    self._connectivity.tree_to_tree = <np.npy_int32*>(cell_adj.data)
    self._connectivity.tree_to_face = <np.npy_int8*>(cell_adj_face.data)

    # self._connectivity.num_corners = 0
    # self._connectivity.corner_to_tree = &self._corner_to_tree[0]
    self._connectivity.ctt_offset = <np.npy_int32*>(corner_to_cell_offset.data)

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
        0,
        <p4est_init_t>_init_quadrant,
        <void*>self )

    self._p4est = p4est
    self._update_iter()

  #-----------------------------------------------------------------------------
  cdef _update_iter(P4est self):
    with nogil:
      p4est_iterate(
        self._p4est,
        NULL,
        <void*>self,
        <p4est_iter_volume_t>_iter_volume,
        NULL,
        NULL )

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
    self._leaf_data = None
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
  def leaf_data(self):
    return self._leaf_data

  #-----------------------------------------------------------------------------
  def leaf_coord(self,
    uv = None,
    idx = None ):
    r"""

    Parameters
    ----------
    uv : None | array of shape = (2,)
      Relative position within each leaf to compute the coordinates, normalized
      to the range [0.0, 1.0] along each edge of the leaf.
      (default: (0.0, 0.0))
    idx : None | any single-dimension ndarray index
      Computes coordinates only for this subset of leaves. (default: slice(None))

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
    # NOTE: the root lvl = 0 is 2**P4EST_MAXLEVEL wide, and all refinement
    # levels are smaller by factors that are powers of 2.
    qwidth = np.left_shift(1, P4EST_MAXLEVEL - info['lvl'].astype(np.int32))

    cell_verts = self.mesh.verts[ self.mesh.cells[ info['cell'] ] ]

    # compute the relative position within the root cell
    UV = np.clip(
      ( info['ij'] + uv[None,:] * qwidth[:,None] ) / P4EST_ROOT_LEN,
      0.0, 1.0)[:,:,None]

    # Compute the coefficients for bilinear interpolation of the root cells
    # vertices absolute position onto the desired position relative to the leaf.
    _UV = 1.0 - UV

    c = np.empty_like(cell_verts)
    c[:,0,0] = _UV[:,0]*_UV[:,1]
    c[:,0,1] = UV[:,0]*_UV[:,1]
    c[:,1,0] = _UV[:,0]*UV[:,1]
    c[:,1,1] = UV[:,0]*UV[:,1]

    # Perform the interpolation
    return np.sum(c * cell_verts, axis = (1,2))

  #-----------------------------------------------------------------------------
  def refine(self,
    recursive = False,
    maxlevel = -1 ):

    # leaf_info = np.copy(self._leaf_info)
    # self._leaf_count = 0

    self._refine(int(recursive), int(maxlevel))
    self._update_iter()

    # self._leaf_info = self._leaf_info[:self._leaf_count]

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

    self._coarsen(int(recursive), int(orphans))
    self._update_iter()

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
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ) with gil:
  pass

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _replace_quadrants(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  int num_outgoing,
  p4est_quadrant_t* outgoing[],
  int num_incoming,
  p4est_quadrant_t* incoming[] ) with gil:

  if num_incoming == 1:
    print(f"coarsening: {cell_idx}")

  else:

    print(f"refining: {cell_idx}")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: these are not class members because they need to match the callback sig.
# but are implemented as though they were by casting the user pointer to be 'self'
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _refine_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ) with gil:

  self = <P4est>p4est.user_pointer
  return 0 < self._leaf_info[quadrant.p.user_long]['refine']

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _coarsen_quadrants(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrants[] ) with gil:

  self = <P4est>p4est.user_pointer
  return 0 > (
    self._leaf_info[quadrants[0].p.user_long]['refine']
    + self._leaf_info[quadrants[1].p.user_long]['refine']
    + self._leaf_info[quadrants[2].p.user_long]['refine']
    + self._leaf_info[quadrants[3].p.user_long]['refine'] )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _weight_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ) with gil:

  self = <P4est>p4est.user_pointer
  return self._leaf_info[quadrant.p.user_long]['weight']

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _iter_volume(
  p4est_iter_volume_info_t* info,
  void *user_data ) with gil:

  cdef p4est_topidx_t cell_idx = info.treeid
  cdef p4est_quadrant_t* quadrant = info.quad
  cdef p4est_ghost_t* ghost = info.ghost_layer

  self = <P4est>user_data

  lc = self._leaf_count

  if lc == len(self._leaf_info):
    # resize array to make room for more leaves as quadrants are refined
    _leaf_info = self._leaf_info
    self._leaf_info = np.zeros((2*len(_leaf_info),), dtype = self._leaf_info.dtype)
    self._leaf_info[lc:] = _leaf_info
    self._leaf_info['cell'][lc:] = -1

  # set where the quadrant data is going to be in the contiguous array
  quadrant.p.user_long = lc

  q = self._leaf_info[lc]
  q['cell'] = cell_idx
  q['lvl'] = quadrant.level
  q['ij'] = (quadrant.x, quadrant.y)
  self._leaf_count += 1

  print(f"+leaf {cell_idx}-{quadrant.level}-{quadrant.x/P4EST_ROOT_LEN:.3f}-{quadrant.y/P4EST_ROOT_LEN:.3f}")
