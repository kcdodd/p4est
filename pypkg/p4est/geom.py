import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexGeometry:
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
  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_linear3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexSpherical(HexGeometry):
  #-----------------------------------------------------------------------------
  def coord(self,
    cell_verts,
    offset ):

    return interp_sphere_to_cart_slerp3(cell_verts, offset)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HexCartesianSpherical(HexGeometry):
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trans_sphere_to_cart(uvr):
  """Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  uvr : array with shape = (..., 3)
    Spherical coordinates are (in order):

    * polar angle (aka. colatitude) [0, pi]
    * azimuthal angle (aka. longitude) [-pi, pi]
    * radius

  Returns
  -------
  xyz : array with shape = (..., 3)
    Cartesian coordinates
  """
  phi = uvr[...,0]
  theta = uvr[...,1]
  r = uvr[...,2]

  xyz = np.zeros_like(uvr)
  xyz[...,0] = r*np.cos(theta)*np.cos(phi)
  xyz[...,1] = r*np.sin(theta)*np.cos(phi)
  xyz[...,2] = r*np.sin(phi)

  return xyz

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trans_cart_to_sphere(xyz):
  """Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  xyz : array with shape = (..., 3)
    Cartesian coordinates

  Returns
  -------
  uvr : array with shape = (..., 3)
    Spherical coordinates are (in order):

    * polar angle (aka. colatitude) [0, pi]
    * azimuthal angle (aka. longitude) [-pi, pi]
    * radius
  """

  x = xyz[...,0]
  y = xyz[...,1]
  z = xyz[...,2]

  r = np.linalg.norm(xyz, axis = -1)
  uvr = np.zeros_like(xyz)

  uvr[...,0] = np.arccos(z / r)
  uvr[...,1] = np.arctan2(y, x)
  uvr[...,2] = r

  return uvr

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_linear(eta, x0, x1):
  """Linear interpolation
  """

  if eta.ndim < x0.ndim:
    eta = eta[...,None]

  return (1.0 - eta) * x0 + eta * x1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_linear2(verts, uv):
  """Bi-linear interpolation
  """

  m = np.prod(verts.shape[-2:])
  s = verts.shape[:-3] + verts.shape[-2:]

  verts = interp_linear(
    eta = uv[...,1],
    x0 = verts[...,0,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:].reshape(-1, m)).reshape(s)

  return interp_linear(
      eta = uv[...,0],
      x0 = verts[...,0,:],
      x1 = verts[...,1,:])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_linear3(verts, uv):
  """Tri-inear interpolation
  """

  m = np.prod(verts.shape[-3:])
  s = verts.shape[:-4] + verts.shape[-3:]

  verts = interp_linear(
    eta = uv[...,2],
    x0 = verts[...,0,:,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

  return interp_linear2(verts = verts, uv = uv)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp(eta, x0, x1):
  """Spherical linear interpolation
  """

  _x0 = np.linalg.norm(x0, axis = -1)
  _x1 = np.linalg.norm(x1, axis = -1)

  cos_theta = np.sum(x0*x1, axis = -1) / (_x0 * _x1)
  theta = np.arccos(cos_theta)
  sin_theta = np.sin(theta)

  c0 = np.sin((1.0 - eta) * theta) / sin_theta
  c1 = np.sin(eta * theta) / sin_theta

  return c0[...,None] * x0 + c1[...,None] * x1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp2(verts, uv):
  """Spherical linear interpolation of quadrilateral vertices
  """

  return interp_slerp(
    eta = uv[...,1],
    x0 = interp_slerp(
      eta = uv[...,0],
      x0 = verts[...,0,0,:],
      x1 = verts[...,0,1,:]),
    x1 = interp_slerp(
      eta = uv[...,0],
      x0 = verts[...,1,0,:],
      x1 = verts[...,1,1,:]) )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp3(verts, uv):
  """Spherical linear interpolation of quadrilateral vertices, assumes third axis
  is 'radial'.
  """

  m = np.prod(verts.shape[-3:])
  s = verts.shape[:-4] + verts.shape[-3:]

  verts = interp_linear(
    eta = uv[...,2],
    x0 = verts[...,0,:,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

  return interp_slerp2(verts, uv)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_sphere_to_cart_slerp2(verts, uv):
  """Spherical linear interpolation applied after transforming to cartesian
  """
  return interp_slerp2(
    verts = trans_sphere_to_cart(verts),
    uv = uv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_sphere_to_cart_slerp3(verts, uv):
  """Spherical linear interpolation applied after transforming to cartesian
  """
  return interp_slerp3(
    verts = trans_sphere_to_cart(verts),
    uv = uv )
