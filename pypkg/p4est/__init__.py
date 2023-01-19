from pathlib import Path
import ctypes

from .geom import (
  trans_sphere_to_cart,
  trans_cart_to_sphere,
  interp_linear,
  interp_linear2,
  interp_linear3,
  interp_slerp,
  interp_slerp2,
  interp_slerp3,
  interp_sphere_to_cart_slerp2,
  interp_sphere_to_cart_slerp3 )
from .mesh import (
  QuadMesh,
  QuadGeometry,
  QuadLinear,
  QuadSpherical,
  QuadCartesianSpherical,
  HexMesh,
  HexGeometry,
  HexLinear,
  HexSpherical,
  HexCartesianSpherical )

from .mesh import (
  QuadAMR,
  HexAMR )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_include():
  return Path(__file__).parent / 'core' / 'include'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_library():
  return Path(__file__).parent / 'core' / 'lib'

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_libsc():
  return ctypes.util.find_library(get_library() / 'libsc')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_libp4est():
  return ctypes.util.find_library(get_library() / 'libp4est')