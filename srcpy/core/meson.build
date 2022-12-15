
cython_gen = generator( cython,
  arguments : [
    '-3',
    '--fast-fail',
    '--output-file', '@OUTPUT@',
    '@INPUT@' ],
  output : '@BASENAME@.c' )

install_files = [
  '__init__.py',
  '__init__.pxd',
  '_p4est.pxd' ]

py3.extension_module(
  '_p4est',
  cython_gen.process(f'_p4est.pyx'),
  c_args: c_args,
  link_args : link_args,
  include_directories: [p4est_inc, py3_inc, np_inc],
  dependencies: py3_deps,
  link_with : p4est_shared_lib,
  install: true,
  # NOTE: this is so the python extensions can find libp4est and libsc at runtime
  install_rpath: '$ORIGIN/lib',
  install_dir: 'p4est/core' )


install_data(
  install_files,
  install_dir: 'p4est/core' )

if get_option('libdir') != 'p4est/core/lib'
  install_symlink(
    'lib',
    install_dir: 'p4est/core',
    pointing_to: get_option('prefix') + '/' + get_option('libdir') )
endif

if get_option('includedir') != 'p4est/core/include'
  install_symlink(
    'include',
    install_dir: 'p4est/core',
    pointing_to: get_option('prefix') + '/' + get_option('includedir') )
endif