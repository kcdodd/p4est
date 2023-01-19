# Enable postponed evaluation of annotations
from __future__ import annotations
try:
  from typing import (
    Optional,
    Union,
    Literal,
    TypeVar,
    NewType )
  from ...typing import N, M, NP, NV, NN, NC
except:
  pass

from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
import numpy as np
from mpi4py import MPI

from ...utils import jagged_array
from ...mesh.quad import QuadMesh
from ...core._info import (
  QuadLocalInfo,
  QuadGhostInfo )
from ...core._adapted import QuadAdapted
from ...core._p4est import P4est

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadAMR(P4est):
  r"""Quadrilateral adaptive mesh refinement

  Parameters
  ----------
  mesh :
    Mesh for the root-level cells.
  max_level :
    (default: -1)
  comm :
    (default: mpi4py.MPI.COMM_WORLD)

  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    mesh : QuadMesh,
    max_level : Optional[int] = None,
    comm : Optional[MPI.Comm] = None ):

    super().__init__(
      mesh = mesh,
      max_level = max_level,
      comm = comm)

  #-----------------------------------------------------------------------------
  @property
  def mesh( self ) -> QuadMesh:
    """Mesh for the root-level cells.
    """
    return self._mesh

  #-----------------------------------------------------------------------------
  @property
  def max_level( self ) -> int:
    return self._max_level

  #-----------------------------------------------------------------------------
  @property
  def comm( self ) -> MPI.Comm:
    return self._comm

  #-----------------------------------------------------------------------------
  @property
  def local(self) -> QuadLocalInfo:
    """Cells local to the process ``comm.rank``.
    """
    return self._local

  #-----------------------------------------------------------------------------
  @property
  def ghost(self) -> jagged_array[NP, QuadGhostInfo]:
    """Cells outside the process boundary (*not* local) that neighbor one or more
    local cells, grouped by the rank of the *ghost's* local process.
    """
    return self._ghost

  #-----------------------------------------------------------------------------
  @property
  def mirror(self) -> jagged_array[NP, np.ndarray[np.dtype[np.integer]]]:
    """Indicies into ``local`` for cells that touch the parallel boundary
    of each rank.
    """
    return self._mirror

  #-----------------------------------------------------------------------------
  def coord(self,
    offset : np.ndarray[(Union[N,Literal[1]], ..., 2), np.dtype[np.floating]],
    where : Union[None, slice, np.ndarray[..., np.dtype[Union[np.integer, bool]]]] = None ) \
    -> np.ndarray[(N, ..., 3), np.dtype[np.floating]]:
    r"""
    Transform to (physical/global) coordinates of a point relative to each cell

    .. math::

      \func{\rankone{r}}{\rankone{q}} =
      \begin{bmatrix}
        \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1}
      \end{bmatrix}

    Parameters
    ----------
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.
      (default: (0.5, 0.5))
    where :
      Subset of cells. (default: :py:obj:`slice(None)`)

    Returns
    -------
    Absolute coordinates at each ``offset``

    """

    return super().coord(
      offset = offset,
      where = where )

  #-----------------------------------------------------------------------------
  def adapt(self) -> tuple[QuadAdapted, QuadAdapted]:
    """Applies refinement, coarsening, and then balances based on ``leaf_info.adapt``.

    Returns
    -------
    ``(refined, coarsened)``
    """

    return super().adapt()
