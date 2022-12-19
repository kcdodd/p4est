import os
import sys
import time
import ruamel.yaml
import shutil
import os.path as osp
from pathlib import Path
from collections.abc import (
  Mapping,
  Sequence,
  Iterable )

cimport numpy as cnp
import numpy as np

from mpi4py import MPI
from mpi4py.MPI cimport MPI_Comm, Comm

from libc.stdlib cimport malloc, free
from libc.string cimport memset

cdef extern from "Python.h":
  const int PyBUF_READ
  const int PyBUF_WRITE
  PyObject *PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _init_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ):
  """
  .. note::

    Only an intermediate callback from p4est, forwards call to bound method
    P4est._init_quadrant to actually handle the action.
  """
  self = <P4est>p4est.user_pointer

  cdef object buffer = <object>PyMemoryView_FromMemory(
    <char*>quadrant.p.user_data,
    self._data_size,
    PyBUF_WRITE )

  data = np.frombuffer(
    buffer,
    dtype = self._dtype,
    count = self._data_count )

  cdef cnp.ndarray[double, ndim = 1] vxyz = np.array([0., 0., 0.], dtype = np._double)

  p4est_qcoord_to_vertex(
    p4est.connectivity,
    cell_idx,
    quadrant.x,
    quadrant.y,
    <double*> vxyz.data )

  # (<P4est>p4est.user_pointer)._init_quadrant(cell_idx, quadrant)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _refine_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ):

  (<P4est>p4est.user_pointer)._refine_quadrant(cell_idx, quadrant)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _coarsen_quadrants(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrants[] ):

  (<P4est>p4est.user_pointer)._coarsen_quadrants(cell_idx, quadrants)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef _weight_quadrant(
  p4est_t* p4est,
  p4est_topidx_t cell_idx,
  p4est_quadrant_t* quadrant ):

  (<P4est>p4est.user_pointer)._weight_quadrant(cell_idx, quadrant)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class P4est:
  #-----------------------------------------------------------------------------
  def __init__(self,
    verts,
    cells,
    shape = None,
    dtype = None,
    cell_adj = None,
    cell_adj_face = None,
    min_quadrants = None,
    min_level = None,
    fill_uniform = None,
    comm = None ):
    """

    Parameters
    ----------
    verts : np.ndarray with shape = (NV, 2 or 3), dtype = np.float64
      Position of each vertex.
    cells : np.ndarray with shape = (NC, 4), dtype = np.int32
      Quadrilateral cells, each defined by the indices of its 4 vertices.

      .. note::

        The order of the vertices must follow the ordering [V00, V01, V10, V11]
        for the desired geometry of the cell.
        E.G For an aligned rectangle: [lower-left, lower-right, upper-left,
        upper-right].

    shape : None | tuple[int]
      The shape of the data stored in each leaf quadrant (default: tuple()).

      .. note::

        A shape = (1,1) result in the same amount of data being stored per-quadrant
        as the default shape = tuple() because an empty shape is also interpreted
        as a single value.

    dtype : None | np.dtype
      The type of the data stored in each leaf quadrant (default: np.float64).

      .. note::

        The top-level dtype may be a composition of other dtypes defined for
        NumPy structured arrasy (https://numpy.org/doc/stable/user/basics.rec.html).
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
    verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 )

    if not (
      verts.ndim == 2
      and verts.shape[1] in [2, 3] ):
      raise ValueError(f"'verts' must have shape (NV, 2 or 3): {verts.shape}")

    if verts.shape[1] == 2:
      # set z coordinates all to zero to get the right shape
      verts = np.append(verts, np.zeros_like(verts[:,:1]), axis = 1)

    #...........................................................................
    cells = cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    if not (
      cells.ndim == 2
      and cells.shape[1] == 4 ):

      raise ValueError(f"'cells' must have shape (NC, 4): {cells.shape}")

    #...........................................................................
    if (cell_adj is None) != (cell_adj_face is None):
      raise ValueError(f"'cell_adj' and 'cell_adj_face' must be specified together")

    if cell_adj is None:
      # build adjacency from per-cell vertex list
      cidx = np.arange(len(cells))

      # faces defined as pairs vertices
      vfaces = np.array([
        # -/+ xfaces
        cells[:,[0,2]],
        cells[:,[1,3]],
        # -/+ yfaces
        cells[:,[0,1]],
        cells[:,[2,3]] ]).transpose((1,0,2))

      # keep track where vertex order is changed to recover orientation
      isort = np.argsort(vfaces, axis = 2)
      reversed = isort[:,:,0] != 0
      vfaces = np.take_along_axis(vfaces, isort, axis = 2)

      # reduce faces down to unique pairs of vertices
      faces, idx, idx_inv = np.unique(
        vfaces.reshape(-1, 2),
        return_index = True,
        return_inverse = True,
        axis = 0 )

      fidx = np.arange(len(faces))

      # indices of unique faces for each cell, according to the vertex order above
      ifaces = idx_inv.reshape(len(cells), 4)

      # reconstructed the 'edges' passing through faces connecting cells
      # i.e. the 'edge list' representation of a graph where cells are the nodes
      dual_edges = -np.ones((len(faces), 2), dtype = np.int32)

      for i in range(4):
        # set cell indices for adjacent pairs
        # FIFO conditional
        j = np.where(dual_edges[ifaces[:,i],0] == -1, 0, 1)
        dual_edges[ifaces[:,i], j] = cidx

      # filter out all faces that don't connect a pair of cells
      # TODO: some kind of default boundary condition?
      m = ~np.any(dual_edges == -1, axis = 1)
      dual_edges = dual_edges[m]
      dual_fidx = fidx[m]

      # indices of first cell
      c0 = dual_edges[:,0]
      # local indices of shared face in first cells
      f0 = np.nonzero(ifaces[c0] == dual_fidx[:,None])[1]

      # indices of second cell
      c1 = dual_edges[:,1]
      # local indices of shared face in second cells
      f1 = np.nonzero(ifaces[c1] == dual_fidx[:,None])[1]

      # relative orientation if one (but not both) are in reversed order
      orientation = (reversed[c0,f0] ^ reversed[c1,f1]).astype(np.int32)

      cell_adj = -np.ones((len(cells), 4), dtype = np.int32)
      cell_adj[c0,f0] = c1
      cell_adj[c1,f1] = c0

      # set the corresponding index of the face and relative orientation to adjacent cell
      cell_adj_face = -np.ones((len(cells), 4), dtype = np.int32)
      cell_adj_face[c0,f0] = f1 + 4*orientation
      cell_adj_face[c1,f1] = f0 + 4*orientation

    #...........................................................................
    cell_adj = np.ascontiguousarray(
      cell_adj,
      dtype = np.int32 )

    if not (
      cell_adj.ndim == 2
      and cell_adj.shape[0] == len(cells)
      and cell_adj.shape[1] == 4 ):

      raise ValueError(f"'cell_adj' must have shape ({len(cells)}, 4): {cells.shape}")

    #...........................................................................
    cell_adj_face = np.ascontiguousarray(
      cell_adj_face,
      dtype = np.int8 )

    if not (
      cell_adj_face.ndim == 2
      and cell_adj_face.shape[0] == len(cells)
      and cell_adj_face.shape[1] == 4 ):

      raise ValueError(f"'cell_adj_face' must have shape ({len(cells)}, 4): {cells.shape}")

    corner_to_tree_offset = [0]

    self._shape = shape
    self._dtype = dtype
    self._data_count = int(np.prod(shape))
    self._data_size = self._data_count * dtype.itemsize


    self._verts = verts
    self._cells = cells
    self._cell_adj = cell_adj
    self._cell_adj_face = cell_adj_face

    # TODO: figure out what are 'corners'
    self._corner_to_tree_offset = np.ascontiguousarray(
      corner_to_tree_offset,
      dtype = np.int32 )

    self._comm = comm

    self._init_c_data(min_quadrants, min_level, fill_uniform)

  #-----------------------------------------------------------------------------
  cdef _init_c_data(
    P4est self,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform ):

    memset(&self._tmap, 0, sizeof(p4est_connectivity_t));

    self._tmap.num_vertices = len(self._verts)
    self._tmap.vertices = <double*>self._verts.data

    self._tmap.num_trees = len(self._cells)
    self._tmap.tree_to_vertex = <np.npy_int32*>self._cells.data
    self._tmap.tree_to_tree = <np.npy_int32*>self._cell_adj.data
    self._tmap.tree_to_face = <np.npy_int8*>self._cell_adj_face.data

    # self._tmap.num_corners = 0
    # self._tmap.corner_to_tree = &self._corner_to_tree[0]
    self._tmap.ctt_offset = <np.npy_int32*>self._corner_to_tree_offset.data

    self._p4est = p4est_new_ext(
      <sc_MPI_Comm> (<Comm>self.comm).ob_mpi,
      &(self._tmap),
      min_quadrants,
      min_level,
      fill_uniform,
      # data_size per quadrant (managed by p4est)
      self._data_size,
      <p4est_init_t>_init_quadrant,
      <void*>self )

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
  cdef _init_quadrant(
    P4est self,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t* quadrant ):

    pass

  #-----------------------------------------------------------------------------
  cdef _refine_quadrant(
    P4est self,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t* quadrant ):
    pass

  #-----------------------------------------------------------------------------
  cdef _coarsen_quadrants(
    P4est self,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t* quadrant[] ):
    pass

  #-----------------------------------------------------------------------------
  cdef _weight_quadrant(
    P4est self,
    p4est_topidx_t cell_idx,
    p4est_quadrant_t* quadrant ):
    pass