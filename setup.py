import setuptools

with open('README.md') as f:
  long_description = f.read()
with open('requirements.txt') as f:
  requirements = [line.strip() for line in f]

setuptools.setup(
  name='hhoppe_utils',
  version='0.0.1',
  author='Hugues Hoppe',
  author_email='hhoppe@gmail.com',
  description='Library of Python tools by Hugues Hoppe',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/hhoppe/hhoppe_utils.git',
  packages=setuptools.find_packages(),
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
  ],
  python_requires='>=3.7',
  install_requires=requirements,
)
