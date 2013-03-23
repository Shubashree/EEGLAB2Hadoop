## Automatically adapted for scipy Oct 05, 2005 by convertcode.py
# Author: Travis Oliphant
#Adapted for use with EEGLAB by matthew burns 2013 

import struct
import sys
import types
import scipy.io
from numpy import *

import warnings
warnings.warn('fopen module is deprecated, please use npfile instead',
              DeprecationWarning, stacklevel=2)

LittleEndian = (sys.byteorder == 'little')

__all__ = ['fopen']

def getsize_type(mtype):
    if mtype in ['B','uchar','byte','unsigned char','integer*1', 'int8']:
        mtype = 'B'
    elif mtype in ['S1', 'char', 'char*1']:
        mtype = 'B'
    elif mtype in ['b', 'schar', 'signed char']:
        mtype = 'b'
    elif mtype in ['h','short','int16','integer*2']:
        mtype = 'h'
    elif mtype in ['H','ushort','uint16','unsigned short']:
        mtype = 'H'
    elif mtype in ['i','int']:
        mtype = 'i'
    elif mtype in ['I','uint','uint32','unsigned int']:
        mtype = 'I'
    elif mtype in ['u4','int32','integer*4']:
        mtype = 'u4'
    elif mtype in ['f','float','float32','real*4', 'real']:
        mtype = 'f'
    elif mtype in ['d','double','float64','real*8', 'double precision']:
        mtype = 'd'
    elif mtype in ['F','complex float','complex*8','complex64']:
        mtype = 'F'
    elif mtype in ['D','complex*16','complex128','complex','complex double']:
        mtype = 'D'
    else:
        mtype = obj2sctype(mtype)

    newarr = empty((1,),mtype)
    return newarr.itemsize, newarr.dtype.char

class fopen(object):
    """Class for reading and writing binary files into numpy arrays.

    Inputs:

      file_name -- The complete path name to the file to open.
      permission -- Open the file with given permissions: ('r', 'H', 'a')
                    for reading, writing, or appending.  This is the same
                    as the mode argument in the builtin open command.
      format -- The byte-ordering of the file:
                (['native', 'n'], ['ieee-le', 'l'], ['ieee-be', 'B']) for
                native, little-endian, or big-endian respectively.

    Attributes (Read only):

      bs -- non-zero if byte-swapping is performed on read and write.
      format -- 'native', 'ieee-le', or 'ieee-be'
      closed -- non-zero if the file is closed.
      mode -- permissions with which this file was opened
      name -- name of the file
    """

#    Methods:
#
#      read -- read data from file and return numpy array
#      write -- write to file from numpy array
#      fort_read -- read Fortran-formatted binary data from the file.
#      fort_write -- write Fortran-formatted binary data to the file.
#      rewind -- rewind to beginning of file
#      size -- get size of file
#      seek -- seek to some position in the file
#      tell -- return current position in file
#      close -- close the file

    def __init__(self,file_name,permission='rb',format='n'):
        if 'b' not in permission: permission += 'b'
        if isinstance(file_name, basestring):
            self.file = file(file_name, permission)
        elif isinstance(file_name, file) and not file_name.closed:
            # first argument is an open file
            self.file = file_name
        else:
            raise TypeError, 'Need filename or open file as input'
        self.setformat(format)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

    def close(self):
        self.file.close()

    def seek(self, *args):
        self.file.seek(*args)

    def tell(self):
        return self.file.tell()

    def raw_read(self, size=-1):
        """Read raw bytes from file as string."""
        return self.file.read(size)

    def raw_write(self, str):
        """Write string to file as raw bytes."""
        return self.file.write(str)

    def setformat(self, format):
        """Set the byte-order of the file."""
        if format in ['native','n','default']:
            self.bs = False
            self.format = 'native'
        elif format in ['ieee-le','l','little-endian','le']:
            self.bs = not LittleEndian
            self.format = 'ieee-le'
        elif format in ['ieee-be','B','big-endian','be']:
            self.bs = LittleEndian
            self.format = 'ieee-be'
        else:
            raise ValueError, "Unrecognized format: " + format
        return

    def read(self,count,stype,rtype=None,bs=None,c_is_b=0):
        """Read data from file and return it in a numpy array.

        Inputs:

          count -- an integer specifying the number of elements of type
                   stype to read or a tuple indicating the shape of
                   the output array.
          stype -- The data type of the stored data (see fwrite method).
          rtype -- The type of the output array.  Same as stype if None.
          bs -- Whether or not to byteswap (or use self.bs if None)
          c_is_b --- If non-zero then the count is an integer
                   specifying the total number of bytes to read
                   (must be a multiple of the size of stype).

        Outputs: (output,)

          output -- a numpy array of type rtype.
        """
        if bs is None:
            bs = self.bs
        else:
            bs = (bs == 1)
        howmany,stype = getsize_type(stype)
        shape = None
        if c_is_b:
            if count % howmany != 0:
                raise ValueError, "When c_is_b is non-zero then " \
                      "count is bytes\nand must be multiple of basic size."
            count = count / howmany
        elif type(count) in [types.TupleType, types.ListType]:
            shape = list(count)
            # allow -1 to specify unknown dimension size as in reshape
            minus_ones = shape.count(-1)
            if minus_ones == 0:
                count = product(shape,axis=0)
            elif minus_ones == 1:
                now = self.tell()
                self.seek(0,2)
                end = self.tell()
                self.seek(now)
                remaining_bytes = end - now
                know_dimensions_size = -product(count,axis=0) * getsize_type(stype)[0]
                unknown_dimension_size, illegal = divmod(remaining_bytes,
                                                         know_dimensions_size)
                if illegal:
                    raise ValueError("unknown dimension doesn't match filesize")
                shape[shape.index(-1)] = unknown_dimension_size
                count = product(shape,axis=0)
            else:
                raise ValueError(
                    "illegal count; can only specify one unknown dimension")
            shape = tuple(shape)
        if rtype is None:
            rtype = stype
        else:
            howmany,rtype = getsize_type(rtype)
        if count == 0:
            return zeros(0,rtype)
        retval = fromfile(self.file, stype,-1,"")
        if shape is not None:             
            retval = resize(retval, shape)
        return retval

    fread = read

    def rewind(self,howmany=None):
        """Rewind a file to its beginning or by a specified amount.
        """
        if howmany is None:
            self.seek(0)
        else:
            self.seek(-howmany,1)

    def size(self):
        """Return the size of the file.
        """
        try:
            sz = self.thesize
        except AttributeError:
            curpos = self.tell()
            self.seek(0,2)
            sz = self.tell()
            self.seek(curpos)
            self.thesize = sz
        return sz