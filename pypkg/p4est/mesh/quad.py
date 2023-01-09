import numpy as np
from ..utils import (
  jagged_array,
  unique_full )
from ..geom import (
  trans_sphere_to_cart )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadMeshBase:
  """Base container for quadrilateral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 2 or 3), dtype = np.float64
    Position of each vertex.
    (AKA :c:var:`p4est_connectivity_t.vertices`)
  cells : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of quadrilateral cells to the indices of their 4 vertices.
    (AKA :c:var:`p4est_connectivity_t.tree_to_vertex`)

    .. code-block::

      cells[:,0,0] -> Vertex(-y, -x)
      cells[:,0,1] -> Vertex(-y, +x)
      cells[:,1,0] -> Vertex(+y, -x)
      cells[:,1,1] -> Vertex(+y, +x)

  cell_adj : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 4 face-adjacent neighbors.
    (AKA :c:var:`p4est_connectivity_t.tree_to_tree`)

    .. code-block::

      cell_adj[:,0,0] -> Cell(-xface)
      cell_adj[:,0,1] -> Cell(+xface)
      cell_adj[:,1,0] -> Cell(-yface)
      cell_adj[:,1,1] -> Cell(+yface)

  cell_adj_face : ndarray with shape = (NC, 2, 2), dtype = np.int8
    Topological order of the faces of each connected cell.
    (AKA :c:var:`p4est_connectivity_t.tree_to_face`)

  cell_nodes : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 4 nodes
    in ``node_cells`` and ``node_cells_inv``,
    ``-1`` used where nodes are not specified.
    (AKA :c:var:`p4est_connectivity_t.tree_to_corner`)

    .. code-block::

      cell_nodes[:,0,0] -> Node(-y, -x)
      cell_nodes[:,0,1] -> Node(-y, +x)
      cell_nodes[:,1,0] -> Node(+y, -x)
      cell_nodes[:,1,1] -> Node(+y, +x)

  node_cells : jagged_array with shape = (NN, *, 1), dtype = np.int32
    Mapping to cells sharing each node, all ``len(node_cells[i]) > 1``.
    (AKA :c:var:`p4est_connectivity_t.corner_to_tree`)
  node_cells_inv : jagged_array with shape = (NN, *, 1), dtype = np.int32
    Mapping to the cell's local vertex {0,1,2,3} in ``cell_nodes`` which maps
    back to the node.
    (AKA :c:var:`p4est_connectivity_t.corner_to_corner`)

    .. code-block::

      nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
      _nodes = cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)]
      valid = nodes == _nodes

  """
  def __init__(self,
    verts,
    cells,
    cell_adj,
    cell_adj_face,
    cell_nodes,
    node_cells,
    node_cells_inv ):

    if not (
      isinstance(node_cells, jagged_array)
      and isinstance(node_cells_inv, jagged_array)
      and node_cells.flat.shape == node_cells_inv.flat.shape
      and (
        node_cells.row_idx is node_cells_inv.row_idx
        or np.all(node_cells.row_idx == node_cells_inv.row_idx) ) ):

      raise ValueError(f"node_cells and node_cells_inv must have the same structure")

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

    self._cell_nodes = np.ascontiguousarray(
      cell_nodes,
      dtype = np.int32 )

    nodes = np.repeat(np.arange(len(node_cells)), node_cells.row_counts)
    _nodes = self._cell_nodes.reshape(-1,4)[(node_cells.flat, node_cells_inv.flat)]

    if not np.all(_nodes == nodes):
      raise ValueError(
        f"'node_cells' and 'node_cells_inv' are not consistent with 'node_cells'")

    # TODO: ensure proper dtype of the jagged_array?
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
class QuadMesh(QuadMeshBase):
  """Conveience constructor for quadrilateral mesh

  Parameters
  ----------
  verts : ndarray with shape = (NV, 2 or 3), dtype = np.float64
    Position of each vertex.
  cells : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.

    .. note::

      The order of the vertices must follow the ordering [[V00, V01], [V10, V11]],
      or [[V0, V1], [V2, V3]], for the desired geometry of the cell.
      E.G For an aligned rectangle: [[lower-left, lower-right], [upper-left,
      upper-right]].

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
      node_cells_inv ) = quad_cell_nodes(cells, vert_nodes)

    independent = self._vert_nodes == -1
    ni = np.count_nonzero(independent)

    full_vert_nodes = np.copy(self._vert_nodes)
    full_vert_nodes[independent] = np.arange(ni) + np.amax(self._vert_nodes) + 1

    ( cell_adj,
      cell_adj_face ) = quad_cell_adjacency(full_vert_nodes[cells])

    super().__init__(
      verts = verts,
      cells = cells,
      cell_adj = cell_adj,
      cell_adj_face = cell_adj_face,
      cell_nodes = cell_nodes,
      node_cells = node_cells,
      node_cells_inv = node_cells_inv )

  #-----------------------------------------------------------------------------
  @property
  def vert_nodes(self):
    return self._vert_nodes

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quad_cell_adjacency(cell_nodes):
  """Derives topological adjacency between quad. cells accross shared faces

  Parameters
  ----------
  cell_nodes : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.

  Returns
  -------
  cell_adj : ndarray with shape (NC, 2, 2) and dtype np.int32
    Topological connectivity to other cells accross each face.
  cell_adj_face : ndarray with shape (NC, 2, 2) and dtype np.int8
    Topological order of the faces of each connected cell.

  """

  nc = len(cell_nodes)
  cidx = np.arange(nc)

  # build adjacency from per-cell vertex list

  # faces defined as 2 nodes each -> (NC,4,2)
  # NOTE: put the new face index as axis = 1, instead of axis = 0
  cell_face_nodes = np.stack((
    # -/+ xfaces
    cell_nodes[:,:,0],
    cell_nodes[:,:,1],
    # -/+ yfaces
    cell_nodes[:,0,:],
    cell_nodes[:,1,:] ),
    axis = 1 )

  sort_idx, unique_mask, unique_idx, inv_idx = unique_full(
    # sort nodes for each face, removing differences in node order
    np.sort(
      # also reshape to a list of 2-node faces (NC,4,2) -> (4*NC,2)
      cell_face_nodes.reshape(4*nc, 2),
      # sort over the 2 nodes of each face
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
  face_cells = np.repeat(sort_idx // 4, repeats).reshape(nf, 2)

  # NOTE: becomes the mapping from each cell to the 6 unique faces
  # (4*NC,) -> (NC,4)
  cell_faces = inv_idx.reshape(nc, 4)


  # indices of first cell of all faces
  c0 = face_cells[:,0]
  # local indices of shared face in first cells
  f0 = np.nonzero(cell_faces[c0] == fidx[:,None])[1]

  # indices of second cell of all faces
  c1 = face_cells[:,1]
  # local indices of shared face in second cells
  f1 = np.nonzero(cell_faces[c1] == fidx[:,None])[1]

  cell_adj = np.empty((nc, 4), dtype = np.int32)
  # default adjacency back onto own index
  cell_adj[:] = np.arange(nc)[:,None]
  # computed adjacency
  cell_adj[c0,f0] = c1
  cell_adj[c1,f1] = c0
  cell_adj = cell_adj.reshape(nc, 2, 2)

  orientation = np.any(cell_face_nodes[c1,f1] != cell_face_nodes[c0,f0], axis = 1)

  # set the corresponding index of the face and relative orientation to adjacent cell
  cell_adj_face = np.empty((nc, 4), dtype = np.int8)
  # default adjacent face is same face
  cell_adj_face[:] = np.arange(4)[None,:]
  # computed adjacent face
  cell_adj_face[c0,f0] = f1 + 4*orientation
  cell_adj_face[c1,f1] = f0 + 4*orientation
  cell_adj_face = cell_adj_face.reshape(nc, 2, 2)

  return cell_adj, cell_adj_face

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quad_cell_nodes(cells, vert_nodes):
  """Derives topological nodes between quad. cells

  Parameters
  ----------
  cells : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Quadrilateral cells, each defined by the indices of its 4 vertices.
  vert_nodes : ndarray with shape = (NV,), dtype = np.int32
    The topological node associated with each vertex.

  Returns
  -------
  vert_nodes : ndarray with shape = (NV,), dtype = np.int32
    Updated ``vert_nodes`` with additional nodes for any detected mesh
    singularities.
  cell_nodes : ndarray with shape = (NC, 2, 2), dtype = np.int32
    Mapping of quadrilateral cells to the indices of their (up to) 4 nodes.
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
  node_cells_inv = (sort_idx % 4).astype(np.int8)

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
  node_keep = (nodes != -1) & ((node_cells_count > 4) | (node_verts_count > 1))

  _node_keep = np.repeat( node_keep, node_cells_count )

  node_cells_count = node_cells_count[node_keep]
  node_cells = node_cells[_node_keep]
  node_cells_inv = node_cells_inv[_node_keep]

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
def unit_square():
  """Factory method to create a unit square

  Returns
  -------
  QuadMesh
  """

  verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0] ])

  cells = np.array([
    [[0, 1], [2, 3]] ])

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cube(length = 1.0):
  """Factory method to create the surface of a cube

  Parameters
  ----------
  length : float

  Returns
  -------
  QuadMesh
  """

  half = 0.5*length

  verts = np.stack(
    np.meshgrid(
      [-half, half],
      [-half, half],
      [-half, half],
      indexing = 'ij'),
    axis = -1).transpose(2,1,0,3).reshape(-1, 3)

  #Cell with vertex ordering [V0, V1] , [V2, V3]
  cells = np.array([
    [[0, 1], [4, 5]], #Origin Cell

    [[1, 3], [5, 7]],  #Right of Origin Cell

    [[2, 0], [6, 4]],  #Left of Origin Cell

    [[3, 2], [7, 6]],  #Opposite of Origin Cell

    [[1, 3], [0, 2]],  #Bottom Cell

    [[5, 7], [4, 6]]])  #Top Cell

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def stack_o_squares():
  verts = np.array([
    [0, 0, 0],
    [1, 0, 0],

    [0, 1, 0],
    [1, 1, 0],

    [0, 2, 0],
    [1, 2, 0],

    [0, 3, 0],
    [1, 3, 0], ],
    dtype = np.float64)

  cells = np.array([
    [[0, 1], [2, 3]],
    [[2, 3], [4, 5]],
    [[4, 5], [6, 7]], ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def star(r1 = 1.0, r2 = 1.5):
  """Factory method to create a 6 point star

  Parameters
  ----------
  r1 : float
    Outer radius of points.

  r2 : float
    Inner radius where points merge.

  Returns
  -------
  QuadMesh
  """

  verts = np.zeros((13, 3), dtype = np.float64)

  i = np.arange(6)
  t = np.pi*i/3

  verts[1::2, 0] = r1*np.cos(t)
  verts[1::2, 1] = r1*np.sin(t)

  verts[2::2, 0] = r2*np.cos(t + np.pi / 6)
  verts[2::2, 1] = r2*np.sin(t + np.pi / 6)

  cells = np.array([
    [[0, 1], [3, 2]],
    [[0, 3], [5, 4]],
    [[5, 6], [0, 7]],
    [[8, 7], [9, 0]],
    [[9, 0], [10, 11]],
    [[12, 1], [11, 0]] ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def periodic_stack():
  """Factory method to create a periodic stack of unit squares

  Parameters
  ----------
  radius : float

  Returns
  -------
  QuadMesh
  """

  verts = np.array([
    [0, 0, 0],
    [1, 0, 0],

    [0, 1, 0],
    [1, 1, 0],

    [0, 2, 0],
    [1, 2, 0],

    [0, 3, 0],
    [1, 3, 0], ],
    dtype = np.float64)

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,-2]] = 0
  vert_nodes[[1,-1]] = 1

  cells = np.array([
    [[0, 1], [2, 3]],
    [[2, 3], [4, 5]],
    [[4, 5], [6, 7]], ],
    dtype = np.int32)

  return QuadMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_golden():
  """Factory method to create an icosahedron using golden ratio

  Parameters
  ----------
  radius : float

  Returns
  -------
  QuadMesh
  """

  c3 = np.cos(np.pi/3)
  s3 = np.sin(np.pi/3)

  verts = np.array([
    [ 0.0 + c3,    s3,  0.0 ],
    [ 1.0 + c3,    s3,  0.0 ],
    [ 2.0 + c3,    s3,  0.0 ],
    [ 3.0 + c3,    s3,  0.0 ],
    [ 4.0 + c3,    s3,  0.0 ],
    [ 0.0,  0.0,  0.0 ],
    [ 1.0,  0.0,  0.0 ],
    [ 2.0,  0.0,  0.0 ],
    [ 3.0,  0.0,  0.0 ],
    [ 4.0,  0.0,  0.0 ],
    [ 5.0,  0.0,  0.0 ],
    [ 0.0 + c3, -  s3,  0.0 ],
    [ 1.0 + c3, -  s3,  0.0 ],
    [ 2.0 + c3, -  s3,  0.0 ],
    [ 3.0 + c3, -  s3,  0.0 ],
    [ 4.0 + c3, -  s3,  0.0 ],
    [ 5.0 + c3, -  s3,  0.0 ],
    [ 0.0 + 2*c3, -2*s3,  0.0 ],
    [ 1.0 + 2*c3, -2*s3,  0.0 ],
    [ 2.0 + 2*c3, -2*s3,  0.0 ],
    [ 3.0 + 2*c3, -2*s3,  0.0 ],
    [ 4.0 + 2*c3, -2*s3,  0.0 ] ])

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,1,2,3,4]] = 0
  # NOTE: this currently crashes during 'balance'
  # vert_nodes[[5,10]] = 2
  # vert_nodes[[11,16]] = 3
  vert_nodes[[17,18,19,20,21]] = 1

  cells = np.array([
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

  return QuadMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron_spherical(radius = 1.0):
  """Factory method to create an icosahedron in spherical coordinates

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

  verts = np.zeros((22,3))

  verts[:5,0] = phi[0]
  verts[:5,1] = theta[:5]

  verts[5:11,0] = phi[1]
  verts[5:11,1] = theta - 0.5*dtheta

  verts[11:17,0] = phi[2]
  verts[11:17,1] = theta

  verts[17:,0] = phi[3]
  verts[17:,1] = theta[:5] + 0.5*dtheta

  verts[:,2] = radius

  # verts = trans_sphere_to_cart(verts)

  vert_nodes = -np.ones(len(verts), dtype = np.int32)
  vert_nodes[[0,1,2,3,4]] = 0
  # NOTE: this currently crashes during 'balance'
  # vert_nodes[[5,10]] = 2
  # vert_nodes[[11,16]] = 3
  vert_nodes[[17,18,19,20,21]] = 1

  cells = np.array([
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

  return QuadMesh(
    verts = verts,
    cells = cells,
    vert_nodes = vert_nodes)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def icosahedron(radius = 1.0):
  """Factory method to create an icosahedron in cartesian coordinates

  Parameters
  ----------
  radius : float
    Outer radius of points

  Returns
  -------
  QuadMesh
  """
  mesh = icosahedron_spherical(radius = radius)

  return QuadMesh(
    verts = trans_sphere_to_cart(mesh.verts),
    cells = mesh.cells,
    vert_nodes = mesh.vert_nodes)
