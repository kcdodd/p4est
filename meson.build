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

mpi_enabled = true
target_pypkg = get_option('pypkg')

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

py3_dep = py3.dependency( required: target_pypkg )

cython = find_program(
  'cython',
  required : target_pypkg )

if target_pypkg and not py3.language_version().version_compare('>= 3.6')
  error('Python 3.6 or newer is required')
endif

# use the console command:
#  date "+%s" to obtain the build number
build_number = run_command(py3,
  [ '-c',
    'import time; print(str(int(time.time())))' ],
  check: true ).stdout().strip()

p4est_config = {
  'PROJECT_VERSION' : meson.project_version(),
  'PROJECT_MAJOR' : meson.project_version().split('.')[0],
  'PROJECT_MINOR' : meson.project_version().split('.')[1],
  'PROJECT_PATCH' : meson.project_version().split('.')[2],
  'P4EST_CC' : '"' + cc.get_id() + '-' + cc.version() + '"',
  'P4EST_CFLAGS' : '""',
  'P4EST_CPP' : '""',
  'P4EST_CPPFLAGS' : '""',
  'P4EST_LDFLAGS' : '""',
  'P4EST_LIBS' : '""' }

check_functions = {
  'P4EST_HAVE_FSYNC' : 'fsync',
  'HAVE_LPTHREAD' : 'pthread_create',
  'HAVE_LUA' : 'lua_createtable',
  'P4EST_HAVE_POSIX_MEMALIGN' : 'posix_memalign' }

check_headers = {
  'P4EST_HAVE_DLFCN_H' : 'dlfcn.h',
  'P4EST_HAVE_INTTYPES_H' : 'inttypes.h',
  'P4EST_HAVE_MEMORY_H' : 'memory.h',
  'P4EST_HAVE_STDINT_H' : 'stdint.h',
  'P4EST_HAVE_STDLIB_H' : 'stdlib.h',
  'P4EST_HAVE_STRINGS_H' : 'strings.h',
  'P4EST_HAVE_STRING_H' : 'string.h',
  'P4EST_HAVE_SYS_STAT_H' : 'sys/stat.h',
  'P4EST_HAVE_SYS_TYPES_H' : 'sys/types.h',
  'P4EST_HAVE_UNISTD_H' : 'unistd.h',
  'P4EST_HAVE_NETINET_IN_H' : 'netinet/in.h',
  'P4EST_HAVE_ARPA_INET_H' : 'arpa/inet.h',
  'P4EST_HAVE_ARPA_NET_H' : 'arpa/inet.h', # ??
  'P4EST_HAVE_WINSOCK2_H' : 'winsock2.h' }

foreach macro, func : check_functions
  if cc.has_function(func)
    p4est_config += { macro : 1 }
  endif
endforeach

foreach macro, header : check_headers
  if cc.has_header(header)
    p4est_config += { macro : 1 }
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

zlib_deps = [
  dependency('Zlib', required : get_option('zlib') ) ]

if zlib_deps[0].found()
  p4est_config += { 'SC_HAVE_ZLIB' : 1 }
endif

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
    required : get_option('mpi') ) ]

mpi_h = cc.has_header('mpi.h',
  required : get_option('mpi') )

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

mpi_enabled = (
  not get_option('mpi').disabled()
  and mpi_h
  and mpi_deps[0].found()
  and mpi_run.compiled() and mpi_run.returncode() == 0 )

if mpi_enabled
  p4est_config += { 'P4EST_ENABLE_MPI' : 1 }
endif

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# check/prepare for building the python extension
# https://mesonbuild.com/Python-module.html
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if target_pypkg

  # ensure mpi4py python package is available
  py3_mpi = py_mod.find_installation(
    'python3',
    modules : ['mpi4py'],
    required: true )

  # NOTE: Strangely, NumPy's pxd file directly referenced the deprecated API,
  # which means that 'cdef ndarray' variables cause compilation errors if
  # 'NPY_NO_DEPRECATED_API' is set.
  # disable NumPy deprecated API
  # numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_16_API_VERSION'
  # c_args += [numpy_nodepr_api]

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

endif

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

if target_pypkg
  subdir('srcpy/core')
endif