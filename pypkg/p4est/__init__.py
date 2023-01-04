from pathlib import Path
import ctypes

from .core import P4est
from .mesh import QuadMesh
from .geom import (
  trans_sphere_to_cart,
  trans_cart_to_sphere,
  interp_slerp,
  interp_slerp_quad,
  interp_sphere_to_cart_slerp )

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