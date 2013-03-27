#!/oasis/scratch/csd181/mdburns/python/bin/python
import sys
import gc
import pickle
import base64
import numpy as np

master={}
gc.disable()
for instr in sys.stdin:

    instr = instr.strip()

    # parse the input we got from mapper.py (limit to one split)
    keystr, valstr = instr.split('\t', 1)
    if keystr=='':
        continue

    sys.stderr.write('reducer: key is ' + keystr +'\n')
    v = pickle.loads(base64.decodestring(valstr))

    id = v['id']
    rov = v['rov']

    sys.stderr.write('reducer: id is ' + id +'\n')
    time_ser_type, num = id.split('.', 1)
    num = int(num)
    sys.stderr.write('reducer: type is ' + time_ser_type +'\n')
    sys.stderr.write('reducer: num is ' + str(num) +'\n')

    #Start a new experiment if it the first record for that experiment
    if not master.has_key(keystr):
        master[keystr]={}
        master[keystr]['ica'] = []
        master[keystr]['raw'] = []

    master[keystr][time_ser_type].append([(num, rov)])


#Sort channels, then emit results as key-val pairs (one per experiment)
for key, types in master.items():
    sorted_array = np.empty((2,len(types['ica'])) + types['ica'][0][0][1].shape)
    for this_type, vals in types.items():
        vals.sort()
        if len(vals) > 0:
            newlist = [this_val[0][1] for this_val in vals]
            newarray = np.empty((0,) + types['ica'][0][0][1].shape)
            newarray = np.append(newarray, newlist, axis=0)

            if this_type=='raw':
                sorted_array[0,:,:,:,:] = newarray
            else:
                sorted_array[1,:,:,:,:] = newarray


    value = base64.b64encode(pickle.dumps(sorted_array, protocol=2))
    num_bytes=sys.getsizeof(value)
    sys.stderr.write('reducer: number of channels in result = '+ str(sorted_array.shape[1]) +' \n')
    sys.stderr.write('reducer: emitting final result for ' + key +': '+ str(num_bytes) + ' bytes total\n')
    print '%s\t%s' % (key, value)
    sys.stderr.write('reducer: '+ key +'... write successful\n')

sys.stderr.write('reducer: finished job\n')

def make_master():
    master={}
    keys = ['X:\\RSVP\\hadoop_input\\exp53_continuous_with_ica', 'X:\\RSVP\\hadoop_input\\exp54_continuous_with_ica']
    ids = ['ica.5', 'ica.1', 'ica.100', 'ica.90', 'raw.5', 'raw.1', 'raw.100', 'raw.90']
    for keystr in keys:
        for id in  ids:
            time_ser_type, num = id.split('.', 1)
            num = int(num)
            sys.stderr.write('reducer: type is ' + time_ser_type +'\n')
            sys.stderr.write('reducer: num is ' + str(num) +'\n')

            #Start a new experiment if it the first record for that experiment
            if not master.has_key(keystr):
                master[keystr]={}
                master[keystr]['ica'] = []
                master[keystr]['raw'] = []

            master[keystr][time_ser_type].append([(num, rov)])

    key, types =  master.items()[0]
    this_type, vals = types.items()[0]



