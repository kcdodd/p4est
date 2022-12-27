import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMesh:
  """

  Parameters
  ----------
  verts : np.ndarray with shape = (NV, 2 or 3), dtype = np.float64
    Position of each vertex.
  cells : np.ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.

    .. note::

      The order of the vertices must follow the ordering [[V00, V01], [V10, V11]],
      or [[V0, V1], [V2, V3]], for the desired geometry of the cell.
      E.G For an aligned rectangle: [[lower-left, lower-right], [upper-left,
      upper-right]].

  cell_adj : None | np.ndarray with shape (NC, 2, 2) and dtype np.int32
    If not given, the adjacency is computed from the cells array.
  cell_adj_face : None | np.ndarray with shape (NC, 2, 2) and dtype np.int8
    If not given, the adjacency is computed from the cells array.

  """
  def __init__(self,
    verts,
    cells,
    cell_adj = None,
    cell_adj_face = None ):

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
      cells.ndim == 3
      and cells.shape[1:] == (2,2) ):

      raise ValueError(f"'cells' must have shape (NC, 2, 2): {cells.shape}")

    #...........................................................................
    if (cell_adj is None) != (cell_adj_face is None):
      raise ValueError(f"'cell_adj' and 'cell_adj_face' must be specified together")

    if cell_adj is None:
      # build adjacency from per-cell vertex list
      cidx = np.arange(len(cells))

      # faces defined as pairs vertices
      vfaces = np.array([
        # -/+ xfaces
        cells[:,[0,1],0],
        cells[:,[0,1],1],
        # -/+ yfaces
        cells[:,0,[0,1]],
        cells[:,1,[0,1]] ]).transpose((1,0,2))

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
      orientation = (reversed[c0,f0] ^ reversed[c1,f1]).astype(np.int8)

      cell_adj = np.empty((len(cells), 4), dtype = np.int32)
      cell_adj[:] = np.arange(len(cells))[:,None]
      cell_adj[c0,f0] = c1
      cell_adj[c1,f1] = c0
      cell_adj = cell_adj.reshape(-1, 2, 2)

      # set the corresponding index of the face and relative orientation to adjacent cell
      cell_adj_face = np.empty((len(cells), 4), dtype = np.int8)
      cell_adj_face[:] = np.arange(4)[None,:]
      cell_adj_face[c0,f0] = f1 + 4*orientation
      cell_adj_face[c1,f1] = f0 + 4*orientation
      cell_adj_face = cell_adj_face.reshape(-1, 2, 2)

    #...........................................................................
    cell_adj = np.ascontiguousarray(
      cell_adj,
      dtype = np.int32 )

    if not (
      cell_adj.ndim == 3
      and cell_adj.shape[0] == len(cells)
      and cell_adj.shape[1:] == (2,2) ):

      raise ValueError(f"'cell_adj' must have shape ({len(cells)}, 2, 2): {cells.shape}")

    #...........................................................................
    cell_adj_face = np.ascontiguousarray(
      cell_adj_face,
      dtype = np.int8 )

    if not (
      cell_adj_face.ndim == 3
      and cell_adj_face.shape[0] == len(cells)
      and cell_adj_face.shape[1:] == (2,2) ):

      raise ValueError(f"'cell_adj_face' must have shape ({len(cells)}, 2, 2): {cells.shape}")

    # TODO: figure out what are 'corners'
    corner_to_cell_offset = None

    if corner_to_cell_offset is None:
      corner_to_cell_offset = [0]

    corner_to_cell_offset = np.ascontiguousarray(
      corner_to_cell_offset,
      dtype = np.int32 )


    #...........................................................................
    self._verts = verts
    self._cells = cells
    self._cell_adj = cell_adj
    self._cell_adj_face = cell_adj_face
    self._corner_to_cell_offset = corner_to_cell_offset

  #-----------------------------------------------------------------------------
  @property
  def verts(self):
    return self._verts

  #-----------------------------------------------------------------------------
  @property
  def cells(self):
    return self._cells

  #-----------------------------------------------------------------------------
  @property
  def cell_adj(self):
    return self._cell_adj

  #-----------------------------------------------------------------------------
  @property
  def cell_adj_face(self):
    return self._cell_adj_face

  #-----------------------------------------------------------------------------
  @property
  def corner_to_cell_offset(self):
    return self._corner_to_cell_offset