#!/bin/bash

#Argument 1 is the local path to hadoop inputs (to be copied to hdfs)
#Argument 2 is the hdfs path to hadoop inputs

################################################################################
START_IDX=44
END_IDX=60
FILESTR=$MY_STOR/data/RSVP/exp?/realtime/exp?_continuous_with_ica
################################################################################

echo 'Starting EEGLAB to hadoop file conversion'

echo -n Copy to HDFS?
read yn

case $yn in
    [Yy]* )rm -r $1/*; echo Deleting local files in $1;;
esac

eeglab2hadoop.py $FILESTR $START_IDX $END_IDX $1 --sequencefile

case $yn in
    [Yy]* ) hdfs -rmr $2/*; hdfs -copyFromLocal $1/* $2; echo Copying to $2;;
esac