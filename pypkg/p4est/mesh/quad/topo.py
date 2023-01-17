import numpy as np
from ...utils import (
  jagged_array,
  unique_full)

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


  independent = (vert_nodes == -1)
  ns = np.count_nonzero(independent)

  if ns > 0:
    # add new nodes for any that need them
    vert_nodes = np.copy(vert_nodes)
    new_nodes = np.arange(ns) + np.amax(vert_nodes) + 1
    vert_nodes[independent] = new_nodes

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
  #_counts = np.diff(node_cells_idx)
  #_nodes = _nodes[_mask]


  # NOTE: this is back to len(nodes), with all 'unused' node counts set to zero
  #node_cells_count = np.zeros((len(nodes),), dtype = np.int32)
  #node_cells_count[_nodes] = _counts

  # Only store nodes that:
  # Are associated with 2 or more vertices ("non-local" connections),
  # or connect 5 or more cells (mesh singularities )

  node_cells = jagged_array(node_cells, node_cells_idx)
  node_cells_inv = jagged_array(node_cells_inv, node_cells_idx)

  return (
    vert_nodes,
    cell_nodes,
    node_cells,
    node_cells_inv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def quad_cell_adj(cell_nodes):
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