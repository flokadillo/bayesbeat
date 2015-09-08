MIREX 2013 Audio Beat Tracking Submission

FK1 Beat.e2013

* Authors:

Florian Krebs <florian.krebs@jku.at>
In case of questions, please contact the author

* Invoke the program with:

matlab -r "compute_beats_mirex_2013('%input','%output'); quit;"

* The program needs the following software to run:

- Matlab (tested on R2010a)
- Compiled viterbi.cpp (see README.md of the bayes_beat class for installation instructions, if you have problems compiling @HMM/viterbi.cpp, you can alternatively set the flag Params.use_mex_viterbi to zero in the file compute_beats_mirex_2013.m)
