#!/bin/bash
ENV=$MY_STOR/python

#Create a virtual environment
pip install virtualenv
virtualenv $ENV --system-site-packages
$ENV/python/bin/activate

#Create lapack
wget http://www.netlib.org/lapack/lapack-3.4.2.tgz
tar -xvf lapack-3.4.2.tgz
rm lapack-3.4.2.tgz
cd lapack-3.4.2
cp make.inc.example make.inc
make blaslib
make

#Install cvxopt
wget http://abel.ee.ucla.edu/src/cvxopt-1.1.5.tar.gz
tar -xvf cvxopt-1.1.5.tar.gz
rm cvxopt-1.1.5.tar.gz
cd cvxopt-1.1.5/src
#setup.py probably needs some tuning... change BLAS/LAPACK dir
python setup.py install

#Install cvxpy
cd $MY_STOR
svn checkout http://cvxpy.googlecode.com/svn/trunk/ cvxpy-read-only
python cvxpy-read-only/setup.py install
python cvxpy-read-only/setup.py test


