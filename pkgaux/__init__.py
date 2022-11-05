import os
import os.path as osp
import shutil

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prep( self, logger ):

  # update version number from file
  with open('VERSION', 'r') as fp:
    self.project.version = fp.read().strip()

  for k in ['openmp']:
    self.meson.options[k] = self.config[k]

  if self.config.debug:

    self.meson.setup_args += [
      '--buildtype', 'debug']

  else:
    self.meson.setup_args += [
      '--buildtype', 'release',
      '--optimization', '3']

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

  # update build number and build (feature) tags
  # NOTE: BUILDNUMBER file doesn't exist until after meson.build is run
  with open(osp.join(self.meson.build_dir, 'BUILDNUMBER'), 'r') as fp:
    build_number = int(fp.read().strip())

  with open(osp.join(self.meson.build_dir, 'BUILDTAGS'), 'r') as fp:
    build_tags = fp.read().strip().split(' ')

  self.binary.build_number = build_number

  self.binary.build_suffix = '_'.join([ tag for tag in build_tags if tag])

