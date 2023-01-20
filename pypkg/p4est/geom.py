# Enable postponed evaluation of annotations
from __future__ import annotations
try:
  from typing import (
    Optional,
    Union,
    Literal,
    TypeVar,
    NewType )
  from .typing import N
except:
  pass

import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trans_sphere_to_cart(
  uvr : np.ndarray[(...,3), np.dtype[np.floating]] ) \
  -> np.ndarray[(...,3), np.dtype[np.floating]]:
  r"""Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  uvr :
    Spherical coordinates are (in order):

    * polar angle (aka. colatitude) :math:`\in [0, \pi]`
    * azimuthal angle (aka. longitude) :math:`\in [-\pi, \pi]`
    * radius

  Returns
  -------
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
def trans_cart_to_sphere(
  xyz : np.ndarray[(...,3), np.dtype[np.floating]] ) \
  -> np.ndarray[(...,3), np.dtype[np.floating]]:
  r"""Transforms coordinates from spherical to cartesian

  Parameters
  ----------
  xyz :
    Cartesian coordinates

  Returns
  -------
  Spherical coordinates are (in order):

  * polar angle (aka. colatitude) :math:`\in [0, \pi]`
  * azimuthal angle (aka. longitude) :math:`\in [-\pi, \pi]`
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
def interp_linear(
  eta : np.ndarray[(...,), np.dtype[np.floating]],
  x0 : np.ndarray[(...,N), np.dtype[np.floating]],
  x1 : np.ndarray[(...,N), np.dtype[np.floating]]) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Linear interpolation

  Parameters
  ----------
  eta :
    Interpolation point :math:`\in [0.0, 1.0]^2`
  x0 :
    Value at :math:`0.0`
  x1 :
    Value at :math:`1.0`
  """

  if eta.ndim < x0.ndim:
    eta = eta[...,None]

  return (1.0 - eta) * x0 + eta * x1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_linear2(
  verts : np.ndarray[(...,2,2,N), np.dtype[np.floating]],
  uv : np.ndarray[(...,2), np.dtype[np.floating]]) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Bi-linear interpolation

  Parameters
  ----------
  verts :
    Values at the 4 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^2`
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
def interp_linear3(
  verts : np.ndarray[(...,2,2,2,N), np.dtype[np.floating]],
  uv : np.ndarray[(...,3), np.dtype[np.floating]] ) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Tri-linear interpolation

  Parameters
  ----------
  verts :
    Values at the 8 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^3`
  """

  m = np.prod(verts.shape[-3:])
  s = verts.shape[:-4] + verts.shape[-3:]

  verts = interp_linear(
    eta = uv[...,2],
    x0 = verts[...,0,:,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

  return interp_linear2(verts = verts, uv = uv)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_slerp(
  eta : np.ndarray[(...,), np.dtype[np.floating]],
  x0 : np.ndarray[(...,N), np.dtype[np.floating]],
  x1 : np.ndarray[(...,N), np.dtype[np.floating]]) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Spherical linear interpolation

  Parameters
  ----------
  eta :
    Interpolation point :math:`\in [0.0, 1.0]^2`
  x0 :
    Value at :math:`0.0`
  x1 :
    Value at :math:`1.0`
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
def interp_slerp2(
  verts : np.ndarray[(...,2,2,N), np.dtype[np.floating]],
  uv : np.ndarray[(...,2), np.dtype[np.floating]]) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Spherical linear interpolation of quadrilateral vertices

  Parameters
  ----------
  verts :
    Values at the 4 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^2`
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
def interp_slerp3(
  verts : np.ndarray[(...,2,2,2,N), np.dtype[np.floating]],
  uv : np.ndarray[(...,3), np.dtype[np.floating]] ) \
  -> np.ndarray[(...,N), np.dtype[np.floating]]:
  r"""Spherical linear interpolation of quadrilateral vertices, assumes third axis
  is 'radial'.

  Parameters
  ----------
  verts :
    Values at the 8 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^3`
  """

  m = np.prod(verts.shape[-3:])
  s = verts.shape[:-4] + verts.shape[-3:]

  verts = interp_linear(
    eta = uv[...,2],
    x0 = verts[...,0,:,:,:].reshape(-1, m),
    x1 = verts[...,1,:,:,:].reshape(-1, m)).reshape(s)

  return interp_slerp2(verts, uv)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_sphere_to_cart_slerp2(
  verts : np.ndarray[(...,2,2,3), np.dtype[np.floating]],
  uv : np.ndarray[(...,2), np.dtype[np.floating]]) \
  -> np.ndarray[(...,3), np.dtype[np.floating]]:
  r"""Spherical linear interpolation applied after transforming to cartesian

  Parameters
  ----------
  verts :
    Values at the 4 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^2`
  """
  return interp_slerp2(
    verts = trans_sphere_to_cart(verts),
    uv = uv )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interp_sphere_to_cart_slerp3(
  verts : np.ndarray[(...,2,2,2,3), np.dtype[np.floating]],
  uv : np.ndarray[(...,3), np.dtype[np.floating]] ) \
  -> np.ndarray[(...,3), np.dtype[np.floating]]:
  r"""Spherical linear interpolation applied after transforming to cartesian

  Parameters
  ----------
  verts :
    Values at the 8 limits of ``uv``.
  uv :
    Interpolation point :math:`\in [0.0, 1.0]^3`
  """
  return interp_slerp3(
    verts = trans_sphere_to_cart(verts),
    uv = uv )
