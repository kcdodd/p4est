



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
cdef LEAF_REFINE = 1 << 31
cdef LEAF_COARSEN = 1 << 32


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef void _init_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ) with gil:
  """
  .. note::

    Only an intermediate callback from p4est, forwards call to bound method
    P4est._init_quadrant to actually handle the action.
  """
  self = <P4est>p4est.user_pointer

  if self._leaf_count == len(self._leaf_info):
    self._leaf_info = np.concatenate([
      self._leaf_info,
      np.zeros_like(self._leaf_info) ])

    self._leaf_info['cell'][self._leaf_count:] = -1

  # set where the quadrant data is going to be in the contiguous array
  quadrant.p.user_long = self._leaf_count

  q = self._leaf_info[self._leaf_count]
  q['cell'] = cell_idx
  q['lvl'] = quadrant.level
  q['i'] = quadrant.x
  q['j'] = quadrant.y
  self._leaf_count += 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _refine_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ) with gil:

  self = <P4est>p4est.user_pointer
  return LEAF_REFINE & self._leaf_info[quadrant.p.user_long]['flags']

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef int _coarsen_quadrants(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrants[] ) with gil:

  self = <P4est>p4est.user_pointer
  return LEAF_COARSEN & (
    self._leaf_info[quadrants[0].p.user_long]['flags']
    & self._leaf_info[quadrants[1].p.user_long]['flags']
    & self._leaf_info[quadrants[2].p.user_long]['flags']
    & self._leaf_info[quadrants[3].p.user_long]['flags'] )

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

  pass

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
    The type of the data stored in each leaf quadrant (default: np.float64).

    .. note::

      The top-level dtype may be a composition of other dtypes defined for
      NumPy structured arrasy (https:#numpy.org/doc/stable/user/basics.rec.html).
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

    if dtype is None:
      dtype = np.float64

    dtype = np.dtype(dtype)

    #...........................................................................
    self._comm = comm
    self._mesh = mesh
    self._shape = shape
    self._dtype = dtype

    self._leaf_dtype = np.dtype([
      ('cell', np.int32),
      ('lvl', np.int8),
      ('i', np.int32),
      ('j', np.int32),
      ('weight', np.int32),
      ('flags', np.int32) ])

    self._leaf_info = np.zeros(
      self._mesh.cells.shape[:1],
      dtype = self._leaf_dtype )

    self._leaf_info['cell'] = -1
    self._leaf_count = 0

    self._init_c_data(min_quadrants, min_level, fill_uniform)

    self._leaf_info = self._leaf_info[:self._leaf_count]
    self._leaf_data = np.zeros(
      (self._leaf_count, *self._shape),
      dtype = self._dtype)

  #-----------------------------------------------------------------------------
  cdef _init_c_data(
    P4est self,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform ):

    memset(&self._connectivity, 0, sizeof(p4est_connectivity_t));

    cdef np.ndarray[double, ndim=2] verts = self._mesh.verts
    cdef np.ndarray[np.npy_int32, ndim=2] cells = self._mesh.cells
    cdef np.ndarray[np.npy_int32, ndim=2] cell_adj = self._mesh.cell_adj
    cdef np.ndarray[np.npy_int8, ndim=2] cell_adj_face = self._mesh.cell_adj_face
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

  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  def leaf_coord(self,
    leaf_info = None,
    offset = None ):

    if leaf_info is None:
      leaf_info = self._leaf_info

    if offset is None:
      offset = np.zeros((2,), dtype = np.float64)

    else:
      offset = np.asarray(offset, dtype = np.float64)

    qwidth = 1 << (P4EST_MAXLEVEL - leaf_info['lvl'])

    verts = self.mesh.verts[ self.mesh.cells[ leaf_info['cell'] ] ]

    wx = np.zeros((2, len(leaf_info)), dtype = np.float64)
    np.clip(
      ( leaf_info['i'] + offset[0] * qwidth ) / P4EST_ROOT_LEN,
      0.0, 1.0,
      out = wx[1])

    wx[0] = 1.0 - wx[1]

    wy = np.zeros((2, len(leaf_info)), dtype = np.float64)
    np.clip(
      ( leaf_info['j'] + offset[1] * qwidth ) / P4EST_ROOT_LEN,
      0.0, 1.0,
      out = wy[1])

    wy[0] = 1.0 - wy[1]

    xyz = np.zeros((len(leaf_info),3), dtype = np.float64)

    for j in range(2):
      yfactor = wy[j]

      for i in range(2):
        xfactor = yfactor * wx[i]
        xyz += xfactor[:,None] * verts[:,2*j+i]

    return xyz

    # P4EST_ROOT_LEN

    # qwidth = 1 << (P4EST_MAXLEVEL - leaf_info['lvl'])

    # for i in range(2):
    #   for j in range(2):
    #     _vcoord = vcoord[i,j]

    #     p4est_qcoord_to_vertex(
    #       &p4est._connectivity,
    #       cell_idx,
    #       quadrant.x + i*qwidth,
    #       quadrant.y + j*qwidth,
    #       <double*>&_vcoord[0] )

