#!/bin/bash
source $CSD181/hadoop/hadoop_shared/hadoop_bashrc.sh
shopt -s expand_aliases

################################################################################
DIR_LOCAL=$MY_STOR/data/RSVP
DIR_HDFS=/user/$USER/RSVP
################################################################################

hdfs_input=$DIR_HDFS/hadoop_input
hdfs_output=$DIR_HDFS/hadoop_output
local_input=$DIR_LOCAL/hadoop_input
local_output=$DIR_LOCAL/hadoop_output

hdfs -rmr $hdfs_output > /dev/null 2>&1
hdfs -mkdir $hdfs_input > /dev/null 2>&1

if [ ! -d $local_output ]; then
	mkdir $local_output
fi

if [ ! -d $local_input ]; then
	mkdir $local_input
fi

#pass y as first arg to recompile data/upload to HDFS (must modify e2p_driver.sh)
case $1 in
    [Yy]* )echo Recompiling local files; e2p_driver.sh $local_input $hdfs_input ;;
esac

#Determine number of map and reduce tasks (max 500 maps)
num=$(hdfs -ls $hdfs_input/|wc -l)
num_seq_files=$(expr $num - 1)
num_chans=$(expr $num_seq_files \* 256)

if [ $(expr $(($num_chans>300))) -eq 1 ]; then
    num_map_tasks=300
else
    num_map_tasks=$num_chans
fi

echo "number of map tasks = $num_map_tasks"

#Submit job
had jar /opt/hadoop/contrib/streaming/hadoop-*streaming*.jar \
-input $hdfs_input/* \
-output $hdfs_output \
-file $MAPR/with_hdfs/mapper.py \
-file $MAPR/with_hdfs/process.py \
-file $MAPR/with_hdfs/reducer.py \
-mapper mapper.py \
-reducer reducer.py \
-inputformat SequenceFileAsTextInputFormat \
-D mapred.map.tasks=$num_map_tasks \
-D mapred.reduce.tasks=$num_seq_files \
-D mapred.tasktracker.map.tasks.maximum=300 \
-D mapred.tasktracker.reduce.tasks.maximum=$num_seq_files \
-D mapred.max.tracker.failures=num_map_tasks \
-D mapred.map.child.java.opts=-Xmx512M \
-D mapred.reduce.child.java.opts=-Xmx2048M \
-D io.sort.mb=2048 \
-D mapred.task.timeout=6000000 \
-D mapred.map.max.attempts=10 \
-D mapred.reduce.max.attempts=10 \
-D mapred.skip.map.max.skip.records=1 \
-D mapred.skip.attempts.to.start.skipping=4 \
-D mapred.skip.mode.enabled=true \

#Copy from HDFS to local
subnumber=$(expr $(ls $local_output -p | grep "/"|wc -l) + 1)
dirname=$local_output/out$subnumber

case $2 in
    [Yy]* )echo Copying results to $dirname; mkdir $dirname; hdfs -copyToLocal $hdfs_output/* $dirname;;
esac

eeglab2hadoop.py $dirname 0 0 0 --hadoop2mat

echo submit_job.sh: done