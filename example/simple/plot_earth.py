from partis.utils import init_logging

init_logging(
  'warning',
  # autodetect color
  with_color = None )

import numpy as np
from scipy.interpolate import (
  RectBivariateSpline)
import sys
import time
from pathlib import Path
from PIL import (
  ImageOps,
  Image )
import pyvista as pv
from p4est import (
  P4est,
  trans_cart_to_sphere )

from p4est.mesh.quad import spherical_cube

from mpi4py.util.dtlib import from_numpy_dtype

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def interpgrid_2D_to_3D():
  im1 = Image.open(Path(__file__).parent / 'earth_no_clouds.jpg')
  im2 = ImageOps.grayscale(im1)
  gray_image = np.array(im2)
  scaled_image = gray_image / 255

  u = np.linspace(0 , np.pi, scaled_image.shape[0])
  v = np.linspace(-np.pi, np.pi, scaled_image.shape[1])
  r = scaled_image

  f = RectBivariateSpline(u , v , r, kx = 1, ky = 1 )

  return f

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mpi_ghost_exchange(grid, local_value):

  # exchange process neighbor information
  sendbuf = np.ascontiguousarray(local_value[grid.rank_mirrors.flat.idx])
  recvbuf = np.empty((len(grid.rank_ghosts.flat),), dtype = local_value.dtype)

  mpi_datatype = from_numpy_dtype(local_value.dtype)

  grid.comm.Alltoallv(
    sendbuf = [
      sendbuf, (
        # counts
        grid.rank_mirrors.row_counts,
        # displs
        grid.rank_mirrors.row_idx[:-1] ),
      mpi_datatype ],
    recvbuf = [
      recvbuf, (
        grid.rank_ghosts.row_counts,
        grid.rank_ghosts.row_idx[:-1] ),
      mpi_datatype ])

  return np.concatenate([local_value, recvbuf])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_grid(grid, scalars = None):
  scale = 0.99
  _scale = 1.0 - scale


  nc = len(grid.leaf_info)
  verts = np.empty((4*nc, 3))
  verts[:nc] = grid.coord(offset = (_scale, _scale))
  verts[nc:2*nc] = grid.coord(offset = (_scale, scale))
  verts[2*nc:3*nc] = grid.coord(offset = (scale, scale))
  verts[3*nc:] = grid.coord(offset = (scale, _scale))

  idx = np.arange(nc)

  faces = np.empty((nc, 5), dtype = np.int32)
  faces[:,0] = 4
  faces[:nc,1] = idx
  faces[:nc,2] = idx + nc
  faces[:nc,3] = idx + 2*nc
  faces[:nc,4] = idx + 3*nc
  pv.set_plot_theme('paraview')
  p = pv.Plotter()
  p.add_mesh(
     pv.PolyData(verts, faces = faces.ravel()),
     scalars = grid.leaf_info.root if scalars is None else scalars,
     show_edges = False,
     line_width = 1,
     point_size = 3 )

  p.show()

  return verts,faces

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mesh = spherical_cube()
f = interpgrid_2D_to_3D()
tol = 0.05


grid = P4est(
  mesh = mesh,
  min_level = 0)

for p in range(4):
  grid.leaf_info.adapt = 1
  grid.adapt()


for r in range(6):

  points = trans_cart_to_sphere(grid.coord(offset = (0.5, 0.5)))
  local_value = f(*points[:,:2].transpose(1,0), grid = False)

  value = mpi_ghost_exchange(grid, local_value)

  value_adj = value[grid.leaf_info.cell_adj]
  d_value = np.abs(local_value[:,None,None,None] - value_adj).max(axis = (1,2,3))

  refine = d_value > tol
  grid.leaf_info.adapt = refine

  if not np.any(refine):
    break

  refined, coarsened = grid.adapt()
  print(f"{grid.comm.rank} refined: {refined.replaced_idx.shape} -> {refined.idx.shape} (total= {len(grid.leaf_info):,})")


points = trans_cart_to_sphere(grid.coord(offset = (0.5,0.5)))
local_value = f(*points[:,:2].transpose(1,0), grid = False)

plot_grid(grid, scalars = local_value)




