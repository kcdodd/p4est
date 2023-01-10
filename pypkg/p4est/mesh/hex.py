import numpy as np
from ..utils import (
  jagged_array,
  unique_full )
from ..geom import (
  trans_sphere_to_cart )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMeshBase:
  """Base container for hexahedral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 3), dtype = np.float64
    Position of each vertex.
    (AKA :c:var:`p8est_connectivity_t.vertices`)
    Indexing is ``[vertex, (x,y,z)]``

  cells : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)

    Indexing is ``[cell, ∓z, ∓y, ∓x]``

    .. code-block::

      cells[:,0,0,0] -> vert(-z, -y, -x)
      cells[:,0,0,1] -> vert(-z, -y, +x)

      cells[:,0,1,0] -> vert(-z, +y, -x)
      cells[:,0,1,1] -> vert(-z, +y, +x)

      cells[:,1,0,0] -> vert(+z, -y, -x)
      cells[:,1,0,1] -> vert(+z, -y, +x)

      cells[:,1,1,0] -> vert(+z, +y, -x)
      cells[:,1,1,1] -> vert(+z, +y, +x)

  cell_adj : ndarray with shape = (NC, 3, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 6 face-adjacent neighbors.
    (AKA :c:var:`p8est_connectivity_t.tree_to_tree`)

    Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

    .. code-block::

      cell_adj[:,0,0] -> xface(-x)
      cell_adj[:,0,1] -> xface(+x)

      cell_adj[:,1,0] -> yface(-y)
      cell_adj[:,1,1] -> yface(+y)

      cell_adj[:,2,0] -> zface(-z)
      cell_adj[:,2,1] -> zface(+z)

  cell_adj_face : ndarray with shape = (NC, 3, 2), dtype = np.int8
    Topological order of the faces of each connected cell.
    (AKA :c:var:`p8est_connectivity_t.tree_to_face`)

    Indexing is ``[cell, (x,y,z), ∓(x|y|z)]``

  cell_edges : ndarray with shape = (NC, 3, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 12 edges
    in ``edge_cells`` and ``edge_cells_``,
    (AKA :c:var:`p8est_connectivity_t.tree_to_edge`)

    Indexing is ``[cell, (x,y,z), ∓(z|z|y), ∓(y|x|x)]``

    .. code-block::

      cell_edges[:,0,0,0] -> xedge(-z, -y)
      cell_edges[:,0,0,1] -> xedge(-z, +y)
      cell_edges[:,0,1,0] -> xedge(+z, -y)
      cell_edges[:,0,1,1] -> xedge(+z, +y)

      cell_edges[:,1,0,0] -> yedge(-z, -x)
      cell_edges[:,1,0,1] -> yedge(-z, +x)
      cell_edges[:,1,1,0] -> yedge(+z, -x)
      cell_edges[:,1,1,1] -> yedge(+z, +x)

      cell_edges[:,2,0,0] -> zedge(-y, -x)
      cell_edges[:,2,0,1] -> zedge(-y, +x)
      cell_edges[:,2,1,0] -> zedge(+y, -x)
      cell_edges[:,2,1,1] -> zedge(+y, +x)

  edge_cells : jagged_array with shape = (NE, *), dtype = np.int32
    Mapping to cells sharing each edge, all ``len(edge_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.edge_to_tree`)

    Indexing is ``[edge, cell]``

  edge_cells_inv : jagged_array with shape = (NE, *), dtype = np.int32
    Mapping to the cell's local edge {0,...11} in ``cell_edges``  which maps
    back to the edge.
    (AKA :c:var:`p8est_connectivity_t.edge_to_edge`)

    Indexing is ``[edge, cell]``

    .. code-block::

      edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
      _edges = cell_edges.reshape(-1,12)[(edge_cells.flat, edge_cells_inv.flat)]
      assert np.all(edges == _edges)

  cell_nodes : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 8 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p8est_connectivity_t.tree_to_corner`)

    Indexing is ``[cell, ∓z, ∓y, ∓x]``

    .. code-block::

      cell_nodes[:,0,0,0] -> node(-z, -y, -x)
      cell_nodes[:,0,0,1] -> node(-z, -y, +x)

      cell_nodes[:,0,1,0] -> node(-z, +y, -x)
      cell_nodes[:,0,1,1] -> node(-z, +y, +x)

      cell_nodes[:,1,0,0] -> node(+z, -y, -x)
      cell_nodes[:,1,0,1] -> node(+z, -y, +x)

      cell_nodes[:,1,1,0] -> node(+z, +y, -x)
      cell_nodes[:,1,1,1] -> node(+z, +y, +x)

  node_cells : jagged_array with shape = (NN, *), dtype = np.int32
    Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p8est_connectivity_t.corner_to_tree`)

    Indexing is ``[node, cell]``

  node_cells_inv : jagged_array with shape = (NN, *), dtype = np.int32
    Mapping to the cell's local vertex {0,...7} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p8est_connectivity_t.corner_to_corner`)

    Indexing is ``[node, cell]``

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,8)[(node_cells.flat, node_cells_inv.flat)]
      assert np.all(nodes == _nodes)

  """
  def __init__(self,
    verts,
    cells,
    cell_adj,
    cell_adj_face,
    cell_edges,
    edge_cells,
    edge_cells_inv,
    cell_nodes,
    node_cells,
    node_cells_inv ):

    self._verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 )

    self._cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 )

    self._cell_adj = np.ascontiguousarray(
      cell_adj,
      dtype = np.int32 )

    self._cell_adj_face = np.ascontiguousarray(
      cell_adj_face,
      dtype = np.int8 )

    self._cell_edges = np.ascontiguousarray(
      cell_edges,
      dtype = np.int32 )

    self._cell_nodes = np.ascontiguousarray(
      cell_nodes,
      dtype = np.int32 )

    #...........................................................................
    # TODO: ensure proper dtype of the jagged_array?
    if not (
      isinstance(edge_cells, jagged_array)
      and isinstance(edge_cells_inv, jagged_array)
      and edge_cells.flat.shape == edge_cells_inv.flat.shape
      and (
        edge_cells.row_idx is edge_cells_inv.row_idx
        or np.all(edge_cells.row_idx == edge_cells_inv.row_idx) ) ):

      raise ValueError(f"edge_cells and edge_cells_inv must have the same structure")

    edges = np.repeat(np.arange(len(edge_cells)), edge_cells.row_counts)
    _edges = cell_edges.reshape(-1,12)[(edge_cells.flat, edge_cells_inv.flat)]

    if not np.all(_edges == edges):
      raise ValueError(
        f"'cell_edges' is not consistent with 'edge_cells' and 'edge_cells_inv'")

    #...........................................................................
    if not (
      isinstance(node_cells, jagged_array)
      and isinstance(node_cells_inv, jagged_array)
      and node_cells.flat.shape == node_cells_inv.flat.shape
      and (
        node_cells.row_idx is node_cells_inv.row_idx
        or np.all(node_cells.row_idx == node_cells_inv.row_idx) ) ):

      raise ValueError(f"node_cells and node_cells_inv must have the same structure")


    nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
    _nodes = self._cell_nodes.reshape(-1,8)[(node_cells.flat, node_cells_inv.flat)]

    if not np.all(_nodes == nodes):
      raise ValueError(
        f"'cell_nodes' is not consistent with 'node_cells' and 'node_cells_inv'.")

    #...........................................................................
    self._edge_cells = edge_cells
    self._edge_cells_inv = edge_cells_inv

    self._node_cells = node_cells
    self._node_cells_inv = node_cells_inv

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
  def cell_edges(self):
    return self._cell_edges

  #-----------------------------------------------------------------------------
  @property
  def edge_cells(self):
    return self._edge_cells

  #-----------------------------------------------------------------------------
  @property
  def edge_cells_inv(self):
    return self._edge_cells_inv

  #-----------------------------------------------------------------------------
  @property
  def cell_nodes(self):
    return self._cell_nodes

  #-----------------------------------------------------------------------------
  @property
  def node_cells(self):
    return self._node_cells

  #-----------------------------------------------------------------------------
  @property
  def node_cells_inv(self):
    return self._node_cells_inv

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexMesh(HexMeshBase):
  """Conveience constructor for hexahedral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 3), dtype = np.float64
    Position of each vertex.

  cells : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of hexahedral cells to the indices of their 8 vertices.
    (AKA :c:var:`p8est_connectivity_t.tree_to_vertex`)

    .. code-block::

      cells[:,0,0,0] -> vert(-z, -y, -x)
      cells[:,0,0,1] -> vert(-z, -y, +x)

      cells[:,0,1,0] -> vert(-z, +y, -x)
      cells[:,0,1,1] -> vert(-z, +y, +x)

      cells[:,1,0,0] -> vert(+z, -y, -x)
      cells[:,1,0,1] -> vert(+z, -y, +x)

      cells[:,1,1,0] -> vert(+z, +y, -x)
      cells[:,1,1,1] -> vert(+z, +y, +x)

  vert_nodes : None | ndarray with shape = (NV,), dtype = np.int32
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

    verts = np.ascontiguousarray(
      verts,
      dtype = np.float64 ).reshape(-1, 3)

    cells = np.ascontiguousarray(
      cells,
      dtype = np.int32 ).reshape(-1, 2, 2, 2)

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
    ( self._vert_nodes,
      cell_nodes,
      node_cells,
      node_cells_inv ) = hex_cell_nodes(cells, vert_nodes)

    independent = self._vert_nodes == -1
    ni = np.count_nonzero(independent)

    full_vert_nodes = np.copy(self._vert_nodes)
    full_vert_nodes[independent] = np.arange(ni) + np.amax(self._vert_nodes) + 1

    full_cell_nodes = full_vert_nodes[cells]

    ( cell_edges,
      edge_cells,
      edge_cells_inv ) = hex_cell_edges(full_cell_nodes)

    ( cell_adj,
      cell_adj_face ) = hex_cell_adjacency(full_cell_nodes)

    super().__init__(
      verts = verts,
      cells = cells,
      cell_adj = cell_adj,
      cell_adj_face = cell_adj_face,
      cell_edges = cell_edges,
      edge_cells = edge_cells,
      edge_cells_inv = edge_cells_inv,
      cell_nodes = cell_nodes,
      node_cells = node_cells,
      node_cells_inv = node_cells_inv )

  #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self):
    return self._vert_nodes


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_nodes(cells, vert_nodes):
  """Derives topological nodes between quad. cells

  Parameters
  ----------
  cells : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Hexahedral cells, each defined by the indices of its 4 vertices.
  vert_nodes : ndarray with shape = (NV,), dtype = np.int32
    The topological node associated with each vertex.

  Returns
  -------
  vert_nodes : ndarray with shape = (NV,), dtype = np.int32
    Updated ``vert_nodes`` with additional nodes for any detected mesh
    singularities.
  cell_nodes : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of hexahedral cells to the indices of their (up to) 8 nodes.
  node_cells : jagged_array, dtype = np.int32
  node_cells_inv : jagged_array, dtype = np.int32
    node_cells_inv.row_idx == node_cells.row_idx
  """

  # count number of cells sharing each vertex to find if there are any singularities
  # that haven't been added to a 'node'
  # NOTE: assumes a vertex is in a cell at most 1 time
  _verts, _count = np.unique(
    cells.ravel(),
    return_counts = True )

  vert_cells_count = np.zeros((len(vert_nodes),), dtype = np.int32)
  vert_cells_count[_verts] = _count
  naked_singularities = (vert_cells_count > 8) & (vert_nodes == -1)
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
  node_cells = (sort_idx // 8).astype(np.int32)

  # also, map which vertex (within the cell)
  # aka. corner_to_corner
  node_cells_inv = (sort_idx % 8).astype(np.int8)

  # Since these are raveled, the offset for each node is needed
  # to reconstruct the individual 'rows' of the jagged array

  # NOTE: This should be equivalent to using np.unique, but..
  # - does not sort the array again.
  # - node_cells_idx is computed directly from intermediate step,
  #   instead of using cumsum to undo the diff
  _nodes = _cell_nodes[sort_idx]

  _mask = np.empty(_nodes.shape, dtype = bool)
  _mask[0] = True
  _mask[1:] = _nodes[1:] != _nodes[:-1]

  # get the indices of the first occurence of each value
  # aka. ctt_offset
  # NOTE: 0 < len(node_cells_idx) <= len(nodes)+1
  node_cells_idx = np.concatenate(np.nonzero(_mask) + ([_mask.size],)).astype(np.int32)

  # the difference between these indices is the count of repeats of the node
  _counts = np.diff(node_cells_idx)
  _nodes = _nodes[_mask]


  # NOTE: this is back to len(nodes), with all 'unused' node counts set to zero
  node_cells_count = np.zeros((len(nodes),), dtype = np.int32)
  node_cells_count[_nodes] = _counts

  # Only store nodes that:
  # Are associated with 2 or more vertices ("non-local" connections),
  # or connect 5 or more cells (mesh singularities )
  node_keep = (nodes != -1) & ((node_cells_count > 8) | (node_verts_count > 1))

  _node_keep = np.repeat( node_keep, node_cells_count )

  node_cells_count = np.ascontiguousarray(node_cells_count[node_keep])
  node_cells = np.ascontiguousarray(node_cells[_node_keep])
  node_cells_inv = np.ascontiguousarray(node_cells_inv[_node_keep])

  # NOTE: this still points to the memoryview of un-raveled cell_nodes
  _cell_nodes[sort_idx[~_node_keep]] = -1

  # recompute the offsets after masking
  node_cells_idx = np.concatenate(([0],np.cumsum(node_cells_count))).astype(np.int32)

  node_cells = jagged_array(node_cells, node_cells_idx)
  node_cells_inv = jagged_array(node_cells_inv, node_cells_idx)

  return (
    vert_nodes,
    cell_nodes,
    node_cells,
    node_cells_inv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_edges(cell_nodes):
  """Derives topological edges shared by cells

  Parameters
  ----------
  cell_nodes : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 8 nodes

  Returns
  -------
  cell_edges : ndarray with shape (NC, 3, 2, 2) and dtype np.int32
  edge_cells : jagged_array
  edge_cells_inv : jagged_array

  """
  nc = len(cell_nodes)
  cidx = np.arange(nc)

  # build edges from per-cell node list (NC,12,2)
  cell_edge_nodes = np.stack(
    ( # x-edges
      cell_nodes[:,0,0,:],
      cell_nodes[:,0,1,:],
      cell_nodes[:,1,0,:],
      cell_nodes[:,1,1,:],
      # y-edges
      cell_nodes[:,0,:,0],
      cell_nodes[:,0,:,1],
      cell_nodes[:,1,:,0],
      cell_nodes[:,1,:,1],
      # z-edges
      cell_nodes[:,:,0,0],
      cell_nodes[:,:,0,1],
      cell_nodes[:,:,1,0],
      cell_nodes[:,:,1,1] ),
    # NOTE: put the new face index as axis = 1, instead of axis = 0,
    axis = 1 )

  # reduce edges down to unique sets of 2 nodes
  sort_idx, unique_mask, unique_idx, inv_idx = unique_full(
    # sort nodes for each face, removing differences in node order
    np.sort(
      # also reshape to a list of 4-node faces (NC,12,2) -> (12*NC,2)
      cell_edge_nodes.reshape(12*nc, 2),
      # sort over the 2 nodes of each edge
      axis = 1 ),
    # unique over all cells
    axis = 0 )

  # mapping of cell to unique edges
  _cell_edges = inv_idx
  # mapping of unique edges to the cells that share it
  edge_cells = (sort_idx // 12).astype(np.int32)
  # mapping to cell's local index of each edge
  edge_cells_inv = (sort_idx % 12).astype(np.int8)

  # keep only edges shared by more than one cell
  edge_cell_counts = np.diff(unique_idx)
  edge_keep = edge_cell_counts > 1
  _edge_keep = np.repeat( edge_keep, edge_cell_counts )

  # set un-used edge indices to -1
  _cell_edges[sort_idx[~_edge_keep]] = -1

  edge_cell_counts = np.ascontiguousarray(edge_cell_counts[edge_keep])
  edge_cells = np.ascontiguousarray(edge_cells[_edge_keep])
  edge_cells_inv = np.ascontiguousarray(edge_cells_inv[_edge_keep])

  # recompute offsets after filtering kept edges
  edge_cells_idx = np.concatenate(([0],np.cumsum(edge_cell_counts))).astype(np.int32)

  # (12*NC,) -> (NC, 3, 2, 2)
  cell_edges = _cell_edges.reshape(nc, 3, 2, 2)

  edge_cells = jagged_array(
    data = edge_cells,
    row_idx = edge_cells_idx )

  edge_cells_inv = jagged_array(
    data = edge_cells_inv,
    row_idx = edge_cells_idx )

  return (
    cell_edges,
    edge_cells,
    edge_cells_inv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_adjacency(cell_nodes):
  """Derives topological adjacency between quad. cells accross shared faces

  Parameters
  ----------
  cell_nodes : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 8 nodes

  Returns
  -------
  cell_adj : ndarray with shape (NC, 3, 2) and dtype np.int32
    Topological connectivity to other cells accross each face.
  cell_adj_face : ndarray with shape (NC, 3, 2) and dtype np.int8
    Topological order of the faces of each connected cell.

  """
  nc = len(cell_nodes)
  cidx = np.arange(nc)

  # build adjacency from per-cell node list (NC,6,2,2) -> (NC,6,4)
  cell_face_nodes = np.stack(
    # 6 faces, defined from 4 nodes each -> (NC,6,2,2)
    ( # -/+ xfaces
      cell_nodes[:,:,:,0],
      cell_nodes[:,:,:,1],
      # -/+ yfaces
      cell_nodes[:,:,0,:],
      cell_nodes[:,:,1,:],
      # -/+ zfaces
      cell_nodes[:,0,:,:],
      cell_nodes[:,1,:,:] ),
    # NOTE: put the new face index as axis = 1, instead of axis = 0,
    axis = 1 ).reshape(nc, 6, 4)


  # reduce faces down to unique sets of 4 nodes
  sort_idx, unique_mask, unique_idx, inv_idx = unique_full(
    # sort nodes for each face, removing differences in node order
    np.sort(
      # also reshape to a list of 4-node faces (NC,6,2,2) -> (6*NC,4)
      cell_face_nodes.reshape(6*nc, 4),
      # sort over the 4 nodes of each face
      axis = 1 ),
    # unique over all cells
    axis = 0 )

  face_counts = np.diff(unique_idx)


  if np.any(face_counts > 2):
    raise ValueError(f"Face shared by more than two cells.")

  # define (arbitrary) indices for list of unique faces
  nf = np.count_nonzero(unique_mask)
  fidx = np.arange(nf)

  # reconstruct, for each face, the (up to) two cells it connects
  # NOTE: each face index appears up to two times in cell_faces,
  # repeat the cell for faces that *don't* connect to another cell
  repeats = np.repeat(3-face_counts, face_counts)
  face_cells = np.repeat(sort_idx // 6, repeats).reshape(nf, 2)

  # NOTE: becomes the mapping from each cell to the 6 unique faces
  # (6*NC,) -> (NC,6)
  cell_faces = inv_idx.reshape(nc, 6)


  # indices of first cell of all faces
  c0 = face_cells[:,0]
  # local indices of shared face in first cells
  f0 = np.nonzero(cell_faces[c0] == fidx[:,None])[1]

  # indices of second cell of all faces
  c1 = face_cells[:,1]
  # local indices of shared face in second cells
  f1 = np.nonzero(cell_faces[c1] == fidx[:,None])[1]

  cell_adj = np.empty((nc, 6), dtype = np.int32)
  # default adjacency back onto own index
  cell_adj[:] = np.arange(nc)[:,None]
  # computed adjacency
  cell_adj[c0,f0] = c1
  cell_adj[c1,f1] = c0
  cell_adj = cell_adj.reshape(nc, 3, 2)

  # Let my_face and other_face
  # be the two face numbers of the connecting trees in 0..5.  Then the first
  # face corner of the lower of my_face and other_face connects to a face
  # corner numbered 0..3 in the higher of my_face and other_face.  The face
  # orientation is defined as this number.

  f0_lower = f0 < f1

  ref_node = np.where(
    f0_lower,
    cell_face_nodes[c0,f0,0],
    cell_face_nodes[c1,f1,0] )

  orientation = np.where(
    f0_lower,
    np.nonzero(cell_face_nodes[c1,f1] == ref_node[:,None])[1],
    np.nonzero(cell_face_nodes[c0,f0] == ref_node[:,None])[1] )

  # set the corresponding index of the face and relative orientation to adjacent cell
  cell_adj_face = np.empty((nc, 6), dtype = np.int8)
  # default adjacent face is same face
  cell_adj_face[:] = np.arange(6)[None,:]
  # computed adjacent face
  cell_adj_face[c0,f0] = f1 + 6*orientation
  cell_adj_face[c1,f1] = f0 + 6*orientation
  cell_adj_face = cell_adj_face.reshape(nc, 3, 2)

  return cell_adj, cell_adj_face


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cube(length = 1.0):
  """Factory method to create the volume of a cube

  Parameters
  ----------
  length : float

  Returns
  -------
  HexMesh
  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  cells = np.arange(8).reshape(1,2,2,2)

  return HexMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_spherical_shell(r1 = 0.5, r2 = 1.0):
  """Factory method to create an icosahedron shell in spherical coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points

  Returns
  -------
  QuadMesh
  """

  c3 = np.cos(np.pi/3)
  s3 = np.sin(np.pi/3)

  theta = np.linspace(-np.pi, np.pi, 6)
  dtheta = theta[1] - theta[0]

  phi = np.array([
    -0.5*np.pi,
    - np.arctan(0.5),
    np.arctan(0.5),
    0.5*np.pi ])

  _verts = np.zeros((22,3))

  _verts[:5,0] = phi[0]
  _verts[:5,1] = theta[:5]

  _verts[5:11,0] = phi[1]
  _verts[5:11,1] = theta - 0.5*dtheta

  _verts[11:17,0] = phi[2]
  _verts[11:17,1] = theta

  _verts[17:,0] = phi[3]
  _verts[17:,1] = theta[:5] + 0.5*dtheta


  verts = np.concatenate([
    _verts + np.array([0.0, 0.0, r1])[None,:],
    _verts + np.array([0.0, 0.0, r2])[None,:] ])

  _vert_nodes = -np.ones(len(_verts), dtype = np.int32)
  _vert_nodes[[0,1,2,3,4]] = 0
  _vert_nodes[[5,10]] = 2
  _vert_nodes[[11,16]] = 3
  _vert_nodes[[17,18,19,20,21]] = 1

  vert_nodes = np.concatenate([
    _vert_nodes,
    np.where(_vert_nodes == -1, _vert_nodes, _vert_nodes + 4) ])

  _cells = np.array([
    [[5,  11], [0,  6]],
    [[11, 17], [6, 12]],
    [[6,  12], [1,  7]],
    [[12, 18], [7, 13]],
    [[7,  13], [2,  8]],
    [[13, 19], [8, 14]],
    [[8,  14], [3,  9]],
    [[14, 20], [9, 15]],
    [[9,  15], [4, 10]],
    [[15, 21], [10, 16]]],
    dtype = np.int32 )

  cells = np.stack([
    _cells,
    _cells + len(_verts)],
    axis = 1 )

  return HexMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_shell(r1 = 0.5, r2 = 1.0):
  """Factory method to create an icosahedron in cartesian coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points

  Returns
  -------
  QuadMesh
  """
  mesh = icosahedron_spherical_shell(r1 = r1, r2 = r2)

  return HexMesh(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes)