HELPERS=~/code/PyEEGLab/helpers
MAPRED=~/code/PyEEGLab/mapreduce
EEGPATH=~/code/PyEEGLab

dos2unix -q $EEGPATH/*
dos2unix -q $HELPERS/*
dos2unix -q $MAPRED/*

chmod --silent -R +x $EEGPATH/*.py 
chmod --silent -R +x $EEGPATH/*.sh
