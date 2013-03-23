#!/home/mdburns/python/bin/python
import eeglab2hadoop

input_str='X:\RSVP\exp?\\realtime\exp?_continuous_with_ica'
substitute=[53];
outpath='X:\RSVP\hadoop_input'
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    global pool
    pool = mp.Pool(mp.cpu_count()-1)

    eeglab2hadoop.compile_data(input_str, substitute, outpath, test_file=True, p=pool)
    pool.close()
    pool.join()
    #eeglab2hadoop.hadoop2mat('U:\data\RSVP\hadoop_output\out1')
