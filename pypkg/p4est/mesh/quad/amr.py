# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal )
  from ...typing import N, M, NP, NV, NN, NC, Where
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
  QuadLocalInfo,
  QuadGhostInfo )
from ...core._adapted import QuadAdapted
from ...core._p4est import P4est
from .base import QuadMesh

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadAMR(P4est):
  r"""Quadrilateral adaptive mesh refinement using p4est

  Parameters
  ----------
  mesh :
    Mesh for the root-level cells.
  max_level :
    (default: -1)
  comm :
    (default: mpi4py.MPI.COMM_WORLD)

  Notes
  -----

  .. subfigure:: AB
    :name: plot_earth
    :subcaptions: above

    .. image:: ../img/amr_earth.png
      :alt: Full view
      :width: 95%

    .. image:: ../img/amr_earth_zoom.png
      :alt: Zoomed view
      :width: 95%

    Earth texture on AMR mesh, refinement set from tolerance on difference in
    value beteen adjacent cells.
  """

  #-----------------------------------------------------------------------------
  def __init__(self,
    mesh : QuadMesh,
    max_level : int = None,
    comm : MPI.Comm = None ):

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
    """Maximum allowed refinement level
    """
    return self._max_level

  #-----------------------------------------------------------------------------
  @property
  def comm( self ) -> MPI.Comm:
    """MPI Communicator
    """
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
    offset : CoordRel,
    where : Where = None ) -> CoordAbs:
    r"""
    Transform to (physical/global) coordinates of a point relative to each cell

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
    refined :
    coarsened :
    """

    return super().adapt()
