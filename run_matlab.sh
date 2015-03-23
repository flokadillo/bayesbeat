#!/bin/bash
# checks if given number (input argument) matches to an existing result directory. if not, simulation is started in the background, and output is written to the logfile.

# Check first argument
if [ -z "$1" ]
  then
    echo "No simulation id supplied - starting from 1"
    COUNTER=1
  else
    COUNTER=$1
fi

# Check second argument
if [ -z "$2" ]
  then
    NSIM=1
  else
    NSIM=$2
fi


#logdir="$HOME/diss/src/matlab/beat_tracking/HMM/results/"
SIMDIR="$HOME/diss/src/matlab/beat_tracking/bayes_beat/results/"

SIMDIRI=${SIMDIR}${COUNTER}

echo "Starting $NSIM simulations"

for (( i=1; i<=$NSIM; i++ ))
do
    while [ -d "$SIMDIRI" ]; 
    do
      let COUNTER=COUNTER+1 
      SIMDIRI=${SIMDIR}${COUNTER}
    done
    mkdir ${SIMDIRI}
    logfln=${SIMDIRI}"/log.txt"
    echo "$i: Write output to ${logfln}"
    matlab -nojvm -nodisplay -singleCompThread -r "BT_Simulation($COUNTER); quit;" 2&> $logfln &
done
