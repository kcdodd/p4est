# Enable postponed evaluation of annotations
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from typing import (
    Optional,
    Union,
    Literal,
    TypeVar,
    NewType )
  from ...typing import N, NP, M, NV, NN, NE, NC, Where
  from .typing import (
    CoordRel,
    CoordAbs)

from collections import namedtuple
from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
import numpy as np
from mpi4py import MPI

from ...utils import jagged_array
from ...core._info import (
  HexLocalInfo,
  HexGhostInfo )
from ...core._adapted import HexAdapted
from ...core._p8est import P8est
from .base import HexMesh

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexAMR(P8est):
  r"""Hexahedral adaptive mesh refinement

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
    mesh : HexMesh,
    max_level : Optional[int] = None,
    comm : Optional[MPI.Comm] = None ):

    super().__init__(
      mesh = mesh,
      max_level = max_level,
      comm = comm)

  #-----------------------------------------------------------------------------
  @property
  def mesh( self ) -> HexMesh:
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
  def local(self) -> HexLocalInfo:
    """Cells local to the process ``comm.rank``.
    """
    return self._local

  #-----------------------------------------------------------------------------
  @property
  def ghost(self) -> jagged_array[NP, HexGhostInfo]:
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
    offset : CoordRel,
    where : Optional[Where] = None ) -> CoordAbs:
    r"""
    Transform to (physical/global) coordinates of a point relative to each cell

    Parameters
    ----------
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.
      (default: (0.5, 0.5, 0.5))
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
  def adapt(self) -> tuple[HexAdapted, HexAdapted]:
    """Applies refinement, coarsening, and then balances based on ``local.adapt``.

    Returns
    -------
    ``(refined, coarsened)``
    """

    return super().adapt()
