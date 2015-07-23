#!/bin/bash
# checks if given number (input argument) matches to an existing result directory. if not, simulation is started in the background, and output is written to the logfile.
echo "Starting matlab..."
matlab -nodisplay -nosplash -r "setenv('LC_ALL','C'); serverFlag = 1; expt_wrapper_inference; exit;"
