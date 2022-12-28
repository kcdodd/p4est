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

  vert_nodes : None | np.ndarray with shape = (NV,), dtype = np.int32
    The topological node associated with each vertex, causing cells to be connected
    by having vertices associated with the same node in addition to directly
    sharing vertices.
    A value of ``-1`` is used to indicate independent vertices.
    If not given, each vertex is assumed to be independent, and cells are only
    connected by shared vertices.

  """
  def __init__(self,
    verts,
    cells,
    vert_nodes = None ):

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
    if vert_nodes is None:
      vert_nodes = -np.ones(len(verts), dtype = np.int32)

    vert_nodes = np.ascontiguousarray(
      vert_nodes,
      dtype = np.int32 )

    if not (
      vert_nodes.ndim == 1
      and vert_nodes.shape[0] == len(verts) ):

      raise ValueError(f"'vert_nodes' must have shape ({len(verts)},): {vert_nodes.shape}")

    _min = np.amin(vert_nodes)
    _max = np.amax(vert_nodes)

    if _min < -1 or _max >= len(verts):
      raise ValueError(f"'vert_nodes' values must be in the range [-1,{len(verts)-1}]: [{_min},{_max}]")

    #...........................................................................
    self._verts = verts
    self._cells = cells

    ( self._cell_adj,
      self._cell_adj_face ) = quad_cell_adjacency(cells)

    ( self._vert_nodes,
      self._cell_nodes,
      self._node_cells,
      self._node_cell_verts,
      self._node_cells_offset ) = quad_cell_nodes(cells, vert_nodes)

    #...........................................................................

    # self._cell_adj = cell_adj
    # self._cell_adj_face = cell_adj_face

    # self._num_nodes_active = num_nodes_active
    # self._cell_nodes = cell_nodes
    # self._node_cells = node_cells
    # self._node_cell_verts = node_cell_verts
    # self._node_cells_offset = node_cells_offset
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
  def cell_nodes(self):
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self):
    return self._vert_nodes

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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quad_cell_adjacency(cells):
  """Computes topological adjacency between quad. cells accross shared faces

  Parameters
  ----------
  cells : np.ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.

  Returns
  -------
  cell_adj : np.ndarray with shape (NC, 2, 2) and dtype np.int32
    Topological connectivity to other cells accross each face.
  cell_adj_face : np.ndarray with shape (NC, 2, 2) and dtype np.int8
    Topological order of the faces of each connected cell.

  """

  cidx = np.arange(len(cells))

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

  return cell_adj, cell_adj_face

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quad_cell_nodes(cells, vert_nodes):
  """Computes topological nodes between quad. cells

  Parameters
  ----------
  cells : np.ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.
  vert_nodes : None | np.ndarray with shape = (NV,), dtype = np.int32
    The topological node associated with each vertex.

  Returns
  -------

  """

  # count number of cells sharing each vertex to find if there are any singularities
  # that haven't been added to a 'node'
  # NOTE: assumes a vertex is in a cell at most 1 time
  _verts, _count = np.unique(
    cells.ravel(),
    return_counts = True )

  vert_cells_count = np.zeros((len(vert_nodes),), dtype = np.int32)
  vert_cells_count[_verts] = _count
  naked_singularities = (vert_cells_count > 4) & (vert_nodes == -1)
  ns = np.count_nonzero(naked_singularities)

  if ns > 0:
    # add new nodes for any that need them
    vert_nodes = np.copy(vert_nodes)
    new_nodes = np.arange(ns) + np.amax(vert_nodes) + 1
    vert_nodes[naked_singularities] = new_nodes

  # get unique nodes, and the number of vertices associated with each one
  nodes, node_verts_count = np.unique(
    vert_nodes,
    return_counts = True )

  if nodes[0] == -1:
    # put independent (-1) nodes at the end for proper indexing
    nodes = np.roll(nodes, -1)
    node_verts_count = np.roll(node_verts_count, -1)

  # aka. tree_to_corner, but as a (NC,2,2) array
  cell_nodes = np.ascontiguousarray(
    vert_nodes[cells],
    dtype = np.int32 )

  _cell_nodes = cell_nodes.ravel()

  # sorting the nodes (in raveled array) would put repeated entries into
  # contiguous groups.
  # Put the *indices* of where the nodes were into these groups using argsort
  # NOTE: the default is 'quicksort', but made explicit in case at some point
  # a stable sort is needed to preserve the relative order of the cell indices
  sort_idx = np.argsort(_cell_nodes, kind = 'quicksort')

  # put independent (-1) nodes at the end for proper indexing
  sort_idx = np.roll(sort_idx, -np.count_nonzero(_cell_nodes == -1))

  # map the indices back to the cell with which the nodes were associated
  # gives a (raveled) jagged array of all the cells associated with
  # each node

  # aka. corner_to_tree
  node_cells = (sort_idx // 4).astype(np.int32)

  # also, map which vertex (within the cell)
  # aka. corner_to_corner
  node_cell_verts = (sort_idx % 4).astype(np.int8)

  # Since these are raveled, the offset for each node is needed
  # to reconstruct the individual 'rows' of the jagged array

  # NOTE: This should be equivalent to using np.unique, but..
  # - does not sort the array again.
  # - node_cells_offset is computed directly from intermediate step,
  #   instead of using cumsum to undo the diff
  _nodes = _cell_nodes[sort_idx]

  _mask = np.empty(_nodes.shape, dtype = bool)
  _mask[0] = True
  _mask[1:] = _nodes[1:] != _nodes[:-1]

  # get the indices of the first occurence of each value
  # aka. ctt_offset
  # NOTE: 0 < len(node_cells_offset) <= len(nodes)+1
  node_cells_offset = np.concatenate(np.nonzero(_mask) + ([_mask.size],)).astype(np.int32)

  # the difference between these indices is the count of repeats of the node
  _counts = np.diff(node_cells_offset)
  _nodes = _nodes[_mask]


  # NOTE: this is back to len(nodes), with all 'unused' node counts set to zero
  node_cells_count = np.zeros((len(nodes),), dtype = np.int32)
  node_cells_count[_nodes] = _counts

  # Only store nodes that:
  # Are associated with 2 or more vertices ("non-local" connections),
  # or connect 5 or more cells (mesh singularities )
  node_keep = (nodes != -1) & ((node_cells_count > 4) | (node_verts_count > 1))

  _node_keep = np.repeat( node_keep, node_cells_count )

  node_cells_count = node_cells_count[node_keep]
  node_cells = node_cells[_node_keep]
  node_cell_verts = node_cell_verts[_node_keep]

  _cell_nodes[sort_idx[~_node_keep]] = -1

  return (
    vert_nodes,
    cell_nodes,
    node_cells,
    node_cell_verts,
    node_cells_offset )