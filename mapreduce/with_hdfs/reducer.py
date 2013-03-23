#!/oasis/scratch/csd181/mdburns/python/bin/python
import sys
import gc
import pickle
import base64

current_key = None
current_count = 0
this_key = None

master={}

for instr in sys.stdin:
    instr = instr.strip()

    # parse the input we got from mapper.py (limit to one split)
    keystr, valstr = instr.split('\t', 1)
    if keystr=='':
        continue

    sys.stderr.write('reducer: key is ' + keystr +'\n')
    v = pickle.loads(base64.decodestring(valstr))

    id = v['id']
    rov_reg, rov_av = v['rov_reg'], v['rov_av']
    sys.stderr.write('reducer: id is ' + id +'\n')
    chan_type, num = id.split('.', 1)

    num = int(num)
    sys.stderr.write('reducer: type is ' + chan_type +'\n')
    sys.stderr.write('reducer: num is ' + str(num) +'\n')

    #Start a new experiment if it the first record for that experiment
    if not master.has_key(keystr):
        this_item = { 'ica':[], 'raw':[] }
        master[keystr]=this_item

    master[keystr][chan_type].append((num, rov_reg, rov_av))

#Sort channels, then emit results as key-val pairs (one per experiment)
for key, types in master.items():
    for this_type, vals in types.items():
        types[this_type] = sorted(vals)

    types['path']=key
    value = base64.b64encode(pickle.dumps(types, protocol=2))
    num_bytes=sys.getsizeof(value)
    sys.stderr.write('reducer: number of items in result = '+ str(len(types['raw'])) +' \n')
    sys.stderr.write('reducer: emitting final result for ' + key +': '+ str(num_bytes) + ' bytes total\n')
    print '%s\t%s' % (key, value)
    sys.stderr.write('reducer: '+ key +'write successful\n')

sys.stderr.write('reducer: good job\n')



