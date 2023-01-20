# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from ...typing import N, M, NV, NN, NE, NC
  from .typing import (
    Cells,
    VertNodes,
    CellNodes,
    CellAdj,
    CellAdjFace,
    NodeCells,
    NodeCellsInv,
    NodeEdgeCounts,
    CellEdges,
    EdgeCells,
    EdgeCellsInv,
    EdgeCellCounts)

import numpy as np
from ...utils import (
  jagged_array,
  unique_full )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_nodes(
  cells : Cells,
  vert_nodes : VertNodes ) \
  -> tuple[VertNodes, CellNodes, NodeCells, NodeCellsInv]:
  """Derives topological nodes between hex. cells

  Parameters
  ----------
  cells :
    Hexahedral cells, each defined by the indices of its 4 vertices.
  vert_nodes :
    The topological node associated with each vertex.

  Returns
  -------
  vert_nodes :
    Updated ``vert_nodes`` with additional nodes for any detected mesh
    singularities.
  cell_nodes :
    Mapping of hexahedral cells to the indices of their (up to) 8 nodes.
  node_cells :
  node_cells_inv :
    ``node_cells_inv.row_idx == node_cells.row_idx``

  """

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

  node_cells = jagged_array(node_cells, node_cells_idx)
  node_cells_inv = jagged_array(node_cells_inv, node_cells_idx)

  return (
    vert_nodes,
    cell_nodes,
    node_cells,
    node_cells_inv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_edges(cell_nodes: CellNodes) \
  -> tuple[CellEdges, EdgeCells, EdgeCellsInv, NodeEdgeCounts, EdgeCellCounts]:
  """Derives topological edges shared by cells

  Parameters
  ----------
  cell_nodes : ndarray with shape = (NC, 2, 2, 2), dtype = np.int32
    Mapping of cells to the indices of their (up to) 8 nodes

  Returns
  -------
  cell_edges :
  edge_cells :
  edge_cells_inv :
  node_edge_counts :
  edge_cell_counts :
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

  # sort nodes for each face, removing differences in node order
  _cell_edge_nodes = np.sort(
    # also reshape to a list of 4-node faces (NC,12,2) -> (12*NC,2)
    cell_edge_nodes.reshape(12*nc, 2),
    # sort over the 2 nodes of each edge
    axis = 1 )

  # reduce edges down to unique sets of 2 nodes
  sort_idx, unique_mask, unique_idx, inv_idx = unique_full(
    _cell_edge_nodes,
    # unique over all cells
    axis = 0 )

  edge_nodes = _cell_edge_nodes[sort_idx[unique_mask]]

  # mapping of cell to unique edges
  _cell_edges = inv_idx
  # mapping of unique edges to the cells that share it
  edge_cells = (sort_idx // 12).astype(np.int32)
  # mapping to cell's local index of each edge
  edge_cells_inv = (sort_idx % 12).astype(np.int8)

  edge_cell_counts = np.diff(unique_idx)
  edge_cells_idx = unique_idx

  # (12*NC,) -> (NC, 3, 2, 2)
  cell_edges = _cell_edges.reshape(nc, 3, 2, 2)

  edge_cells = jagged_array(
    data = edge_cells,
    row_idx = edge_cells_idx )

  edge_cells_inv = jagged_array(
    data = edge_cells_inv,
    row_idx = edge_cells_idx )

  # compute the number of edges incident on each node
  nodes, node_edge_counts = np.unique(edge_nodes, return_counts = True)

  if nodes[0] == -1:
    node_edge_counts = np.ascontiguousarray(node_edge_counts[1:])

  return (
    cell_edges,
    edge_cells,
    edge_cells_inv,
    edge_cell_counts,
    node_edge_counts )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hex_cell_adj(cell_nodes: CellNodes) -> tuple[CellAdj, CellAdjFace]:
  """Derives topological adjacency between hex. cells accross shared faces

  Parameters
  ----------
  cell_nodes :
    Mapping of cells to the indices of their (up to) 8 nodes

  Returns
  -------
  cell_adj :
    Topological connectivity to other cells accross each face.
  cell_adj_face :
    Topological order of the faces of each connected cell

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