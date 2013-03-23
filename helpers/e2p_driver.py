#!/home/mdburns/python/bin/python
import eeglab2hadoop

input_str='X:\RSVP\exp?\\realtime\exp?_continuous_with_ica'
substitute=[53];
outpath='X:\RSVP\hadoop'
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    #eeglab2hadoop.compile_data(input_str, substitute, outpath, test_file=True)
    eeglab2hadoop.hadoop2mat('U:\data\RSVP\hadoop_output\out1')
