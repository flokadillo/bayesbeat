#!/bin/bash
# checks if given number (input argument) matches to an existing result directory. if not, simulation is started in the background, and output is written to the logfile.

#logdir="$HOME/diss/src/matlab/beat_tracking/HMM/results/"
SIMDIR="$HOME/diss/src/matlab/beat_tracking/bayes_beat/results/"

COUNTER=$1
SIMDIRI=${SIMDIR}${COUNTER}

while [ -d "$SIMDIRI" ]; 
do
  let COUNTER=COUNTER+1 
  SIMDIRI=${SIMDIR}${COUNTER}
done
mkdir ${SIMDIRI}
logfln=${SIMDIRI}"/log.txt"
echo "Write output to ${logfln}"
matlab -nojvm -nodisplay -singleCompThread -r "BT_Simulation($COUNTER); quit;" 2&> $logfln &
