# Enable postponed evaluation of annotations
from __future__ import annotations
from partis.utils import TYPING

if TYPING:
  from typing import (
    Union,
    Literal )
  from ...typing import N, M, NV, NN, NE, NC, Where
  from .typing import (
    CoordRel,
    CoordAbs,
    CoordAbsJac,
    CoordGeom)

import numpy as np
from ...geom import (
  interp_linear3,
  interp_sphere_to_cart_slerp3,
  interp_slerp3 )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexGeometry:
  """Base interface for defining 3D geometry on hexahedrons
  """

  #-----------------------------------------------------------------------------
  def __init__(self):
    pass

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts : CoordGeom,
    offset : CoordRel ) -> CoordAbs:
    r"""Transform to (physical/global) coordinates of a point relative to each cell

    Parameters
    ----------
    cell_verts :
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.


    Returns
    -------
    coord: Absolute coordinates at each ``offset``
    """

    raise NotImplementedError()

  #-----------------------------------------------------------------------------
  def coord_jac(self,
    cell_verts : CoordGeom,
    offset : CoordRel ) -> CoordAbsJac:
    r"""Jacobian of the absolute coordinates w.r.t local coordinates

    Parameters
    ----------
    cell_verts :
    offset :
      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.


    Returns
    -------
    jac: Jacobian at each ``offset``
    """
    raise NotImplementedError()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexLinear(HexGeometry):
  """Linear 3D geometry
  """

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts : CoordGeom,
    offset : CoordRel ) -> CoordAbs:

    return interp_linear3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexSpherical(HexGeometry):
  """Spherical 3D geometry
  """

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts : CoordGeom,
    offset : CoordRel ) -> CoordAbs:

    return interp_sphere_to_cart_slerp3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexCartesianSpherical(HexGeometry):
  """Spherical 3D geometry in cartesian coordinates

  Parameters
  ----------
  origin :
    Center of sphere in absolute coordinates
  """

  #-----------------------------------------------------------------------------
  def __init__(self, origin : tuple[float,float,float] = None):
    super().__init__()

    if origin is None:
      origin = (0.0, 0.0, 0.0)

    self.origin = np.array(origin, dtype = np.float64)

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts : CoordGeom,
    offset : CoordRel ) -> CoordAbs:

    return interp_slerp3(cell_verts - self.origin, offset) + self.origin
