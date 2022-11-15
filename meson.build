# meson setup build
# ninja -C build -j 12
project( 'p4est',
  'c',
  version : configure_file(input: 'VERSION', output: 'VERSION', copy: true),
  license : 'GPL',
  meson_version: '>= 0.61.0',
  default_options : [
    'c_std=gnu11' ] )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# set desired compiler options, if available
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
cc = meson.get_compiler('c')

compiler_options = [
  '-Wno-unknown-warning-option',
  '-Wno-unused-variable',
  '-Wno-unused-parameter',
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wfatal-errors',
  '-Wnull-dereference',
  '-Warray-bounds=2',
  '-Werror-implicit-function-declaration',
  '-Werror=return-type',
  '-Werror=discarded-qualifiers' ]

foreach w : compiler_options
  if cc.has_argument(w)
    add_global_arguments(w, language : 'c')
  endif
endforeach

# enable POSIX features
c_args = [
  '-D_POSIX_C_SOURCE=200809L' ]

link_args = []


pkgconfig = import('pkgconfig')

# Used to run some of the general build scripts
py_mod = import('python')
py3 = py_mod.find_installation('python3', required: true)

message('Python: ' + py3.language_version())

py3_dep = py3.dependency( required: true )

cython = find_program(
  'cython',
  required : true )

if not py3.language_version().version_compare('>= 3.6')
  error('Python 3.6 or newer is required')
endif

# use the console command:
#  date "+%s" to obtain the build number
build_number = run_command(py3,
  [ '-c',
    'import time; print(str(int(time.time())))' ],
  check: true ).stdout().strip()

check_functions = {
  'adler32_combine' : 'P4EST_HAVE_ZLIB' }

check_headers = {
  'netinet/in.h' : 'P4EST_HAVE_NETINET_IN_H',
  # TODO: P4EST_HAVE_ARPA_INET_H ???
  'arpa/inet.h' : 'P4EST_HAVE_ARPA_NET_H',
  'winsock2.h' : 'P4EST_HAVE_WINSOCK2_H',
  'unistd.h' : 'P4EST_HAVE_UNISTD_H' }

foreach func, macro : check_functions
  if cc.has_function(func)
    c_args += [ '-D@0@=1'.format(macro) ]
  endif
endforeach

foreach header, macro : check_headers
  if cc.has_header(header)
    c_args += [ '-D@0@=1'.format(macro) ]
  endif
endforeach

# P4_TO_P8
# P4EST_WITH_METIS
# P4EST_ENABLE_DEBUG
# P4EST_ENABLE_MPI
# P4EST_ENABLE_MPIIO
# P4EST_VTK_BINARY
# P4EST_VTK_COMPRESSION
# P4EST_VTK_DOUBLES

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External library dependencies
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# NOTE: the math library is made optional for systems without an independent math library
m_deps = [
  cc.find_library('m', required : false ) ]


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Check for enabling MPI
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# NOTE: there is currently a bug in mesonbuild detection of MPI as a named
# dependency. This simply checks that the mpi library can be found.
# TODO: revert when meson merges the open issue for MPI dependency checks


# NOTE: this is manually checking for the mpicc and mpicxx wrappers, even if
# they were not manually specified through the command line.
# https://github.com/mesonbuild/meson/pull/7373

mpi_deps = [
  cc.find_library('mpi',
    required : true ) ]

mpi_h = cc.has_header('mpi.h',
  required : true )

mpi_cc_code = '''
#include <mpi.h>

int main(void) {
  MPI_Init( 0, 0 );
  MPI_Comm comm = MPI_COMM_WORLD;
  int size;
  MPI_Comm_size( comm, &size );
  MPI_Finalize();
  return 0;
}
'''

mpi_run = cc.run(
  mpi_cc_code,
  dependencies: mpi_deps,
  name : 'MPI program' )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# check/prepare for building the python extension
# https://mesonbuild.com/Python-module.html
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ensure mpi4py python package is available
py3_mpi = py_mod.find_installation(
  'python3',
  modules : ['mpi4py'],
  required: true )

# disable NumPy deprecated API
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

c_args += [numpy_nodepr_api]

# Python include directory
incdir_py3 = run_command(py3,
  [ '-c',
    'from sysconfig import get_paths; print(get_paths()["include"])' ],
  check: true ).stdout().strip()

message('Python includes: ' + incdir_py3)
py3_inc = include_directories(incdir_py3)

# NumPy include directory
incdir_numpy = run_command(py3,
  [ '-c',
    'import numpy; print(numpy.get_include())' ],
  check: true ).stdout().strip()

message('NumPy includes: ' + incdir_numpy)
np_inc = include_directories(incdir_numpy)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Build SC
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

sc_proj = subproject('sc')
sc_dep = sc_proj.get_variable('sc_shared_dep')

py3_deps = [py3_dep,] + mpi_deps + m_deps + sc_dep
sc_deps = [ sc_dep ]


p4est_inc = include_directories('src')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# configure files
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

build_number_file = configure_file(
  input: 'BUILDNUMBER.in',
  output: 'BUILDNUMBER',
  configuration : {
    'buildnumber' : build_number })

subdir('src')
