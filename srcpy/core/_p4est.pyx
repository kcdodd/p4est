import os
import sys
import time
import ruamel.yaml
import shutil
import os.path as osp
from pathlib import Path
from collections.abc import (
  Mapping,
  Sequence )

cimport numpy as np
import numpy as np

from mpi4py import MPI
from mpi4py.MPI cimport MPI_Comm, Comm

from libc.stdlib cimport malloc, free
from libc.string cimport memset


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cdef class p4est:
  #-----------------------------------------------------------------------------
  def __init__(self,
    min_quadrants = None,
    min_level = None,
    fill_uniform = None,
    comm = None ):

    if min_quadrants is None:
      min_quadrants = 0

    if min_level is None:
      min_level = 0

    if fill_uniform is None:
      fill_uniform = False

    if comm is None:
      comm = MPI.COMM_WORLD

    vertices = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0] ]

    tree_to_vertex = [
      [0, 1, 2, 3] ]

    tree_to_tree = [
      [0, 0, 0, 0] ]

    tree_to_face = [
      [0, 1, 2, 3] ]

    corner_to_tree_offset = [0]

    self._vertices = np.ascontiguousarray(
      vertices,
      dtype = np.float64 )

    if not (
      self._vertices.ndim == 2
      and self._vertices.shape[1] == 3 ):

      raise ValueError()

    self._tree_to_vertex = np.ascontiguousarray(
      tree_to_vertex,
      dtype = np.int32 )

    if not (
      self._tree_to_vertex.ndim == 2
      and self._tree_to_vertex.shape[1] == 4 ):

      raise ValueError()

    self._tree_to_tree = np.ascontiguousarray(
      tree_to_tree,
      dtype = np.int32 )

    if not (
      self._tree_to_tree.ndim == 2
      and self._tree_to_tree.shape[0] == self._tree_to_vertex.shape[0]
      and self._tree_to_tree.shape[1] == 4 ):

      raise ValueError()

    self._tree_to_face = np.ascontiguousarray(
      tree_to_face,
      dtype = np.int8 )

    if not (
      self._tree_to_face.ndim == 2
      and self._tree_to_face.shape[0] == self._tree_to_vertex.shape[0]
      and self._tree_to_face.shape[1] == 4
      # The values for tree_to_face are 0..7
      and not np.any(self._tree_to_face & ~0x07) ):

      raise ValueError()

    self._corner_to_tree_offset = np.ascontiguousarray(
      corner_to_tree_offset,
      dtype = np.int32 )

    self._comm = comm

    self._init_c_data(min_quadrants, min_level, fill_uniform)

  #-----------------------------------------------------------------------------
  cdef _init_c_data(
    p4est self,
    p4est_locidx_t min_quadrants,
    int min_level,
    int fill_uniform ):

    memset(&self._tmap, 0, sizeof(p4est_connectivity_t));

    cdef np.ndarray arr = np.zeros((3,))
    self._tmap.num_vertices = arr.shape[0]
    self._tmap.vertices = <double*>self._vertices.data

    self._tmap.num_trees = self._tree_to_tree.shape[0]
    self._tmap.tree_to_tree = <np.npy_int32*>self._tree_to_tree.data

    self._tmap.tree_to_vertex = <np.npy_int32*>self._tree_to_vertex.data
    self._tmap.tree_to_face = <np.npy_int8*>self._tree_to_face.data

    # self._tmap.num_corners = 0
    # self._tmap.corner_to_tree = &self._corner_to_tree[0]
    self._tmap.ctt_offset = <np.npy_int32*>self._corner_to_tree_offset.data

    print(self._tmap.num_vertices, self._tmap.num_trees)

    self._p4est = p4est_new_ext(
      <sc_MPI_Comm> (<Comm>self.comm).ob_mpi,
      &(self._tmap),
      min_quadrants,
      min_level,
      fill_uniform,
      0,
      NULL,
      NULL )

  #-----------------------------------------------------------------------------
  def __dealloc__(self):
    """Deallocate c-level system
    """
    self.free()

  #-----------------------------------------------------------------------------
  def free(self):
    p4est_destroy(self._p4est)

  #-----------------------------------------------------------------------------
  def __enter__(self):
    return self

  #-----------------------------------------------------------------------------
  def __exit__(self, type, value, traceback):
    return False

  #-----------------------------------------------------------------------------
  @property
  def comm( self ):
    return self._comm