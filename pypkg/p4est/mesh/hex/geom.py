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
    cell_verts,
    offset ):
    r"""Transform to (physical/global) coordinates of a point relative to each cell

    .. math::

      \func{\rankone{r}}{\rankone{q}} =
      \begin{bmatrix}
        \func{\rankzero{x}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
        \func{\rankzero{y}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2} \\
        \func{\rankzero{z}}{\rankzero{q}_0, \rankzero{q}_1, \rankzero{q}_2}
      \end{bmatrix}

    Parameters
    ----------
    cell_verts : numpy.ndarray
      shape = (..., 2, 2, 2, 3)

    offset : numpy.ndarray
      shape = (..., 3)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.


    Returns
    -------
    coord: array of shape = (..., 3)
    """

    raise NotImplementedError()

  #-----------------------------------------------------------------------------
  def coord_jac(self,
    cell_verts,
    offset ):
    r"""Jacobian of the absolute coordinates w.r.t local coordinates

    .. math::

      \ranktwo{J}_\rankone{r} = \nabla_{\rankone{q}} \rankone{r} =
      \begin{bmatrix}
        \frac{\partial x}{\partial q_0} & \frac{\partial x}{\partial q_1} & \frac{\partial x}{\partial q_2} \\
        \frac{\partial y}{\partial q_0} & \frac{\partial y}{\partial q_1} & \frac{\partial y}{\partial q_2} \\
        \frac{\partial z}{\partial q_0} & \frac{\partial z}{\partial q_1} & \frac{\partial z}{\partial q_2}
      \end{bmatrix}

    Parameters
    ----------
    cell_verts : numpy.ndarray
      shape = (..., 2, 2, 2, 3)

    offset : numpy.ndarray
      shape = (..., 3)

      Relative coordinates from each cell origin to compute the coordinates,
      normalized :math:`\rankone{q} \in [0.0, 1.0]^3` along each edge of the cell.


    Returns
    -------
    coord_jac: numpy.ndarray
      shape = (..., 3, 3)
    """
    raise NotImplementedError()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexLinear(HexGeometry):
  """Linear 3D geometry
  """

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_linear3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexSpherical(HexGeometry):
  """Spherical 3D geometry
  """

  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_sphere_to_cart_slerp3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexCartesianSpherical(HexGeometry):
  """Spherical 3D geometry in cartesian coordinates
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

    return interp_slerp3(cell_verts - self.origin, offset) + self.origin
