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

  nodes : None | np.ndarray with shape = (NV,), dtype = np.int32
    The topological node associated with each vertex. If not given this
    defaults to ``arange(NV)`` giving a 1:1 association of vertices to nodes.
  cell_adj : None | np.ndarray with shape (NC, 2, 2) and dtype np.int32
    Topological connectivity to other cells accross each face.
    If not given, the adjacency is computed from the cells array.
  cell_adj_face : None | np.ndarray with shape (NC, 2, 2) and dtype np.int8
    Topological order of the faces of each connected cell.
    If not given, the adjacency is computed from the cells array.

  """
  def __init__(self,
    verts,
    cells,
    nodes = None,
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
    cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    if not (
      cells.ndim == 3
      and cells.shape[1:] == (2,2) ):

      raise ValueError(f"'cells' must have shape (NC, 2, 2): {cells.shape}")

    _min = np.amin(cells)
    _max = np.amax(cells)

    if _min < 0 or _max >= len(verts):
      raise ValueError(f"'cells' values must be in the range [0,{len(verts)-1}]: [{_min},{_max}]")

    #...........................................................................
    if nodes is None:
      nodes = np.arange(len(verts))

    nodes = np.ascontiguousarray(
      nodes,
      dtype = np.int32 )

    if not (
      nodes.ndim == 1
      and nodes.shape[0] == len(verts) ):

      raise ValueError(f"'nodes' must have shape ({len(verts)},): {nodes.shape}")

    _min = np.amin(nodes)
    _max = np.amax(nodes)

    if _min < 0 or _max >= len(verts):
      raise ValueError(f"'nodes' values must be in the range [0,{len(verts)-1}]: [{_min},{_max}]")

    #...........................................................................
    if (cell_adj is None) != (cell_adj_face is None):
      raise ValueError(f"'cell_adj' and 'cell_adj_face' must be specified together")

    cidx = np.arange(len(cells))

    if cell_adj is None:
      # build adjacency from per-cell vertex list

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


    #...........................................................................
    # aka. tree_to_corner, but as a (NC,2,2) array
    cell_nodes = np.ascontiguousarray(
      nodes[cells],
      dtype = np.int32 )

    _cell_nodes = cell_nodes.ravel()

    print('cell_nodes')
    print(cell_nodes)
    print(_cell_nodes)

    node_idx = np.argsort(_cell_nodes)
    print('node_idx')
    print(node_idx)

    node_cells_count = np.zeros((len(nodes),), dtype = np.int32)
    _nodes, counts = np.unique(
      _cell_nodes,
      return_counts = True)

    node_cells_count[_nodes] = counts

    print('node_cells_count')
    print(node_cells_count)
    print()

    # aka. corner_to_tree
    node_cells = node_idx // 4
    print('node_cells')
    print(node_cells)

    # aka. corner_to_corner
    node_cell_verts = (node_idx % 4).astype(np.int8)
    print('node_cell_verts')
    print(node_cell_verts)

    # aka. ctt_offset
    node_cells_offset = np.empty((len(nodes)+1,), dtype = np.int32)
    node_cells_offset[0] = 0
    node_cells_offset[1:] = np.cumsum(node_cells_count)
    print('node_cells_offset')
    print(node_cells_offset)
    print()

    # Only store nodes shared by vertices that connect two or more cells
    # node_verts_active = np.zeros((len(nodes),), dtype = bool)
    _nodes, idx_inv, vert_counts = np.unique(
      nodes,
      return_inverse = True,
      return_counts = True)

    print('idx_inv')
    print(idx_inv)

    print('vert_counts')
    print(vert_counts)

    # node_verts_active[idx_inv] = np.repeat(vert_counts > 1, vert_counts)
    node_multi = node_cells_count > 1
    # node_multi[_nodes[vert_counts <= 1]] = False

    # print('node_verts_active')
    # print(node_verts_active)

    print('node_multi')
    print(node_multi)

    # aka num_corners
    num_nodes_active = np.count_nonzero(node_multi)
    print('num_nodes_active', num_nodes_active)

    _node_multi = np.repeat( node_multi, node_cells_count )

    node_cells_count = node_cells_count[node_multi]
    node_cells = node_cells[_node_multi]
    node_cell_verts = node_cell_verts[_node_multi]

    _cell_nodes[node_idx[~_node_multi]] = -1

    print('m -> cell_nodes')
    print(cell_nodes)

    print('m -> node_cells')
    print(node_cells)


    #...........................................................................
    self._verts = verts
    self._cells = cells
    self._nodes = nodes

    self._cell_adj = cell_adj
    self._cell_adj_face = cell_adj_face

    self._num_nodes_active = num_nodes_active
    self._cell_nodes = cell_nodes
    self._node_cells = node_cells
    self._node_cell_verts = node_cell_verts
    self._node_cells_offset = node_cells_offset
    # self._node_verts_active = node_verts_active

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
  def num_nodes_active(self):
    return self._num_nodes_active

  #-----------------------------------------------------------------------------
  # @property
  # def node_verts_active(self):
  #   return self._node_verts_active

  #-----------------------------------------------------------------------------
  @property
  def node_cells_offset(self):
    return self._node_cells_offset

  #-----------------------------------------------------------------------------
  @property
  def node_cells(self):
    return self._node_cells

  #-----------------------------------------------------------------------------
  @property
  def node_cell_verts(self):
    return self._node_cell_verts
