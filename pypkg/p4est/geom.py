import numpy as np

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
  return (1.0 - eta)[...,None] * x0 + eta[...,None] * x1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_bilinear(verts, uv):
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
def interp_trilinear(verts, uv):
  """Tri-inear interpolation
  """

  m = np.prod(verts.shape[-3:])
  s = verts.shape[:-4] + verts.shape[-3:]

  verts = interp_linear(
    eta = uv[...,2],
    x0 = verts[...,0,:,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

  return interp_bilinear(verts = verts, uv = uv)

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
def interp_slerp_quad(verts, uv):
  """Spherical linear interpolation of quadrilateral vertices, assumes third axis
  is 'radial'.
  """

  if uv.shape[-1] == 3:
    m = np.prod(verts.shape[-3:])
    s = verts.shape[:-4] + verts.shape[-3:]

    verts = interp_linear(
      eta = uv[...,2],
      x0 = verts[...,0,:,:,:].reshape(-1, m),
      x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

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
def interp_sphere_to_cart_slerp(verts, uv):
  """Spherical linear interpolation applied after transforming to cartesian
  """
  return interp_slerp_quad(
    verts = trans_sphere_to_cart(verts),
    uv = uv )
