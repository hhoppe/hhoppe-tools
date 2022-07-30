"""Setup for pip package."""

import pathlib
import setuptools

NAME = 'hhoppe-tools'

def get_version(package=None):
  if package is None:
    package, = setuptools.find_packages()
  path = pathlib.Path(__file__).resolve().parent / package / '__init__.py'
  for line in path.read_text().splitlines():
    if line.startswith("__version__ = '"):
      _, version, _ = line.split("'")
      return version
  raise RuntimeError(f'Unable to find version string in {path}.')


def get_requirements():
  path = pathlib.Path('.') / 'requirements.txt'
  return [line.strip() for line in path.read_text().splitlines()
          if not (line.isspace() or line.startswith('#'))]


setuptools.setup(
  name=NAME,
  version=get_version(),
  author='Hugues Hoppe',
  author_email='hhoppe@gmail.com',
  description='Library of Python tools by Hugues Hoppe',
  long_description=pathlib.Path('README.md').read_text(),
  long_description_content_type='text/markdown',
  url=f'https://github.com/hhoppe/{NAME}.git',
  packages=setuptools.find_packages(),
  package_data={
    package: ['py.typed'] for package in setuptools.find_packages()
  },
  classifiers=[
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ],
  python_requires='>=3.7',
  install_requires=get_requirements(),
)
