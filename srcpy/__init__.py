from .core import P4est
from .mesh import QuadMesh

def get_include():
  import os.path as osp
  return osp.join(osp.dirname(__file__), 'core', 'include')

def get_library():
  import os.path as osp
  return osp.join(osp.dirname(__file__), 'core', 'lib')