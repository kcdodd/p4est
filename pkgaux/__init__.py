import os
import os.path as osp
import shutil

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prep( self, logger ):

  # update version number from file
  with open('VERSION', 'r') as fp:
    self.project.version = fp.read().strip()


  cc = os.environ.get('CC', '')
  cxx = os.environ.get('CXX', '')

  # attempt to set the MPI compiler before running meson
  if 'mpi' not in cc or 'mpi' not in cxx:
    # NOTE: checks that CC is not manually set to specific MPI compiler

    mpicc = shutil.which('mpicc')
    mpicxx = shutil.which('mpicxx')

    if mpicc and mpicxx:

      os.environ['CC'] = mpicc
      os.environ['CXX'] = mpicxx

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dist_binary_prep( self, logger ):
  build_dir = self.targets[0].build_dir 

  # update build number and build (feature) tags
  # NOTE: BUILDNUMBER file doesn't exist until after meson.build is run
  with open( build_dir / 'BUILDNUMBER', 'r') as fp:
    build_number = int(fp.read().strip())

  self.binary.build_number = build_number

