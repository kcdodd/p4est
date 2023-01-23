from ..info import Info, InfoField
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadLocalInfo(Info):
  """AMR cells owned by local process
  """
  __slots__ = ()

  root = InfoField(tuple(), np.int32)
  """The index of the original root level mesh.cells
  """

  idx = InfoField(tuple(), np.int32)
  """unique local index
  """

  level = InfoField(tuple(), np.int8)
  """Refinement level
  """

  origin = InfoField((2,), np.int32)
  """Normalized coordinate of the leaf's origin relative to the root cell
  stored as integer units to allow exact arithmetic:
  ``0 -> 0.0``
  ``2**max_level -> 1.0``
  To get positions from this origin, the relative width of the leaf can be
  computed from the refinement level:
  ``2**(max_level - level) -> 1.0/2**level``

  .. note::

    This results in higher precision than a normalized 32bit float,
    since a single-precision float only has 24bits for the fraction.
    Floating point arithmetic involving the normalized coordinates should
    use 64bit (double) precision to avoid loss of precision.
  """

  weight = InfoField(tuple(), np.int32)
  """Computational weight of the leaf used for load partitioning among
  processors.
  """


  adapt = InfoField(tuple(), np.int8)
  """A flag used to control refinement (>0) and coarsening (<0)
  """

  cell_adj = InfoField((2,2,2), np.int32)
  """Indices of up to 6 unique adjacent cells, ordered as:

  .. code-block::

       |110 | 111|
    ---+----+----+---
    001|         |011
    ---+         +---
    000|         |010
    ---+----+----+---
       |100 | 101|

  Indexing: ``[(x-normal, y-normal), (-face, +face), (-half, +half)]``
  If ``cell_adj[i,j,0] == cell_adj[i,j,1]``, then both halfs are shared with
  the same cell (of equal or greater size).
  Otherwise each half is shared with different cells of 1/2 size.
  """

  cell_adj_face = InfoField((2,2), np.int8)
  """Connecting face {0..7} of the adjacent cell

  .. code-block::

       |   11   |
    ---+--------+---
       |        |
     00|        |01
       |        |
    ---+--------+---
       |   10   |

  .. note::

    These do not have to be specified for each half like cell_adj, since
    the adjacent face is the same for both halves since both neighbors must be
    part of the same tree
  """

  cell_adj_subface = InfoField((2,2), np.int8)
  """Connect sub-face, {0,1} in the case where the adjacent cell is twice the size
  """

  cell_adj_order = InfoField((2,2), np.int8)
  """relative ordering {-1, 1} of the adjacent cell
  """


  cell_adj_level = InfoField((2,2), np.int8)
  """relative level {-1, 0, 1} of the adjacent cell
  """

  cell_adj_rank = InfoField((2,2,2), np.int32)
  """MPI process rank of adjacent cell
  """

  cell_nodes = InfoField((2,2), np.int32)
  """The nodes that are inside of a cell
  """

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadGhostInfo(Info):
  """AMR cells directly neighboring those of local process
  """
  __slots__ = ()

  rank = InfoField(tuple(), np.int32)
  """The rank of the owning process
  """

  root = InfoField(tuple(), np.int32)
  """The index of the original root level mesh.cells
  """

  idx = InfoField(tuple(), np.int32)
  """unique local index
  """

  level = InfoField(tuple(), np.int8)
  """Refinement level
  """

  origin = InfoField((2,), np.int32)
  """Normalized coordinate of the leaf's origin relative to the root cell.
  """