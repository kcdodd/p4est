# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal )
  from ...typing import (
    NP,
    NTX,
    NRX,
    NL,
    NG,
    NM,
    NAM,
    NAF,
    NAC,
    Where )
  from .typing import (
    CoordRel,
    CoordAbs)

from collections.abc import (
  Iterable,
  Sequence,
  Mapping )
import numpy as np
from mpi4py import MPI

from ...utils import jagged_array, InfoUpdate
from .info import (
  QuadLocalInfo,
  QuadGhostInfo )
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
    value between adjacent cells.
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
  def local(self) -> QuadLocalInfo[NL]:
    """Cells local to the process ``comm.rank``.
    """
    return self._local

  #-----------------------------------------------------------------------------
  @property
  def ghost(self) -> jagged_array[NP, QuadGhostInfo[NG]]:
    """Cells outside the process boundary (*not* local) that neighbor one or more
    local cells, grouped by the rank of the *ghost's* local process.

    Examples
    --------

    .. code-block:: python

      from mpi4py.util.dtlib import from_numpy_dtype

      # exchange process neighbor information
      sendbuf = np.ascontiguousarray(local_value[grid.mirror.flat])
      recvbuf = np.empty((len(grid.ghost.flat),), dtype = local_value.dtype)

      mpi_datatype = from_numpy_dtype(local_value.dtype)

      grid.comm.Alltoallv(
        sendbuf = [
          sendbuf, (
            # counts
            grid.mirror.row_counts,
            # displs
            grid.mirror.row_idx[:-1] ),
          mpi_datatype ],
        recvbuf = [
          recvbuf, (
            grid.ghost.row_counts,
            grid.ghost.row_idx[:-1] ),
          mpi_datatype ])

      value = np.concatenate([local_value, recvbuf])

      value_adj = value[grid.local.cell_adj]
      d_value = np.abs(local_value[:,None,None,None] - value_adj).max(axis = (1,2,3))
    """
    return self._ghost

  #-----------------------------------------------------------------------------
  @property
  def mirror(self) -> jagged_array[NP, np.ndarray[NM, np.dtype[np.integer]]]:
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
  def adapt(self) -> tuple[
    InfoUpdate[QuadLocalInfo[NAM], QuadLocalInfo[NAM]],
    InfoUpdate[QuadLocalInfo[NAF,2,2], QuadLocalInfo[NAF]],
    InfoUpdate[QuadLocalInfo[NAC], QuadLocalInfo[NAC,2,2]]]:
    """Applies refinement, coarsening, and then balances based on ``leaf_info.adapt``.

    Returns
    -------
    moved :
    refined :
    coarsened :

    Examples
    --------

    .. code-block:: python

      moved, refined, coarsened = grid.adapt()

      # update local values that have changed position, refined, or coarsened
      _local_value = -np.ones(len(grid.local), dtype = local_value.dtype)
      _local_value[moved.dst.idx] = local_value[moved.src.idx]
      _local_value[refined.dst.idx] = local_value[refined.src.idx,None,None]
      _local_value[coarsened.dst.idx] = local_value[coarsened.src.idx].mean(axis = (1,2))
      local_value = _local_value
    """

    return self._adapt()

  #-----------------------------------------------------------------------------
  def partition(self) -> tuple[
    jagged_array[NP, QuadLocalInfo[NTX]],
    jagged_array[NP, QuadLocalInfo[NRX]]]:
    """Applies partitioning based on ``local.weight``.

    Returns
    -------
    send_to :
    receive_from :

    Examples
    --------

    .. code-block:: python

      from mpi4py.util.dtlib import from_numpy_dtype

      tx, rx = grid.partition()

      # exchange values that have been moved between ranks
      sendbuf = np.ascontiguousarray(local_value[tx.flat.idx])
      recvbuf = np.empty((len(rx.flat),), dtype = local_value.dtype)

      mpi_datatype = from_numpy_dtype(local_value.dtype)

      grid.comm.Alltoallv(
        sendbuf = [
          sendbuf, (
            # counts
            tx.row_counts,
            # displs
            tx.row_idx[:-1] ),
          mpi_datatype ],
        recvbuf = [
          recvbuf, (
            rx.row_counts,
            rx.row_idx[:-1] ),
          mpi_datatype ])

      # handle elements that only moved on local rank
      tx0 = tx.row_idx[grid.comm.rank]
      tx1 = tx.row_idx[grid.comm.rank+1]

      rx0 = rx.row_idx[grid.comm.rank]
      rx1 = rx.row_idx[grid.comm.rank+1]

      recvbuf[rx0:rx1] = sendbuf[tx0:tx1]

      local_value = recvbuf
    """

    return self._partition()
