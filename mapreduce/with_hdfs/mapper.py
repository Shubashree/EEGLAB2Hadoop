#!/oasis/scratch/csd181/mdburns/python/bin/python
import sys
import pickle
import base64
from process import process

##import multiprocessing
##from multiprocessing import Pool
EPOCH_LENGTH=.875
EPOCH_OFFSET=.125
NUM_FOLDS=5

for instr in sys.stdin:
    this_key=''
    sys.stderr.write('mapper: begin receiving data\n')
    instr = instr.strip()
    keystr, valstr = instr.split('\t', 1)
    sys.stderr.write('mapper: key_string ' + keystr + '\n')

    this_key, this_id = keystr.split('.', 1)
    sys.stderr.write('mapper: key is ' + keystr +'\n')
    sys.stderr.write('mapper: this_key is ' + this_key +'\n')
    sys.stderr.write('mapper: this_id is ' + this_id +'\n')

    v = pickle.loads(base64.decodestring(valstr))
    y = v[0].reshape((-1,1))
    eeg = pickle.loads(v[1])

    try:
        reduction_variance_reg, reduction_variance_av = process(y, eeg, EPOCH_LENGTH, EPOCH_OFFSET, NUM_FOLDS)
        result = {'id':this_id, 'rov_av':reduction_variance_av, 'rov_reg':reduction_variance_reg }
    except:
        sys.stderr.write('mapper: process failed\n')
        continue

    this_val = base64.b64encode(pickle.dumps(result, protocol=2))

    if this_key != '':
        print '%s\t%s' % (this_key, this_val)

sys.stderr.write('mapper:  good job\n')