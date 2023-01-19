# Enable postponed evaluation of annotations
from __future__ import annotations
try:
  from typing import (
    Optional,
    Union,
    Literal,
    TypeVar,
    NewType )
  from ...typing import N
except:
  pass

import numpy as np
from ...geom import (
  interp_linear2,
  interp_sphere_to_cart_slerp2,
  interp_slerp2 )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadGeometry:
  """Base interface for defining 2D geometry on quadrilaterals
  """

  #-----------------------------------------------------------------------------
  def __init__(self):
    pass

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts : np.ndarray[(Union[N,Literal[1]], ..., 2,2,3), np.dtype[np.floating]],
    offset : np.ndarray[(Union[N,Literal[1]], ..., 2), np.dtype[np.floating]] ) \
      -> np.ndarray[(N, ..., 3), np.dtype[np.floating]]:
    r"""Transform to (physical/global) coordinates of a point relative to each cell

    .. math::

      \func{\rankone{r}}{\rankone{q}} =
      \begin{bmatrix}
        \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1} \\
        \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1}
      \end{bmatrix}

    Parameters
    ----------
    cell_verts :
    offset :

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.

    Returns
    -------
    Absolute coordinates at each ``offset``
    """

    raise NotImplementedError()

  #-----------------------------------------------------------------------------
  def coord_jac(self,
    cell_verts : np.ndarray[(Union[N,Literal[1]], ..., 2,2,3), np.dtype[np.floating]],
    offset : np.ndarray[(Union[N,Literal[1]], ..., 2), np.dtype[np.floating]] ) \
      -> np.ndarray[(N, ..., 3,2), np.dtype[np.floating]]:
    r"""Jacobian of the absolute coordinates w.r.t local coordinates

    .. math::

      \ranktwo{J}_\rankone{r} = \nabla_{\rankone{q}} \rankone{r} =
      \begin{bmatrix}
        \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} \\
        \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} \\
        \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1}
      \end{bmatrix}

    Parameters
    ----------
    cell_verts :
    offset :

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^2` along each edge of the cell.

    Returns
    -------
    Jacobian at each ``offset``
    """
    raise NotImplementedError()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadLinear(QuadGeometry):
  """Linear 2D geometry
  """
  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_linear2(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadSpherical(QuadGeometry):
  """Spherical 2D geometry
  """

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_sphere_to_cart_slerp2(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class QuadCartesianSpherical(QuadGeometry):
  """Spherical 2D geometry in cartesian coordinates
  """

  #-----------------------------------------------------------------------------
  def __init__(self, origin = None):
    super().__init__()

    if origin is None:
      origin = (0.0, 0.0, 0.0)

    self.origin = np.array(origin, dtype = np.float64)

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_slerp2(cell_verts - self.origin, offset) + self.origin
