#!/usr/bin/env python
from distutils.core import setup

setup(name='sequencefile',
      version='0.1',
      description='Python Hadoop I/O Utilities',
      license="Apache Software License 2.0 (ASF)",
      author='Matteo Bertozzi',
      author_email='theo.bertozzi@gmail.com',
      url='http://hadoop.apache.org',
      packages=["sequencefile", 'sequencefile.util', 'sequencefile.io', 'sequencefile.io.compress']
     )