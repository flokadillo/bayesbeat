# The bayes_beat package

This file describes how to use the bayes-beat tracker class. The model is based on a paper originally published by Whiteley et al. in 2006 [1]. The model was adapted to audio data by Krebs et al. 2013 [2].

The bayes-beat class is best explored by looking at the examples folder. So far, there are four examples present:

1. Compute beats using a pretrained HMM model
2. Compute beats using a pretrained PF model
3. Learn the HMM observation model parameters from training data

If you want to quickly apply the beat tracker to some audio input, have a look at the `./bin` folder where you will find several beat trackers that use the *bayes_beat* base class.

[1] Whiteley, Nick, Ali Taylan Cemgil, and Simon J. Godsill. "Bayesian Modelling of Temporal Structure in Musical Audio." Proc. of the 14th International Conference on Music Information Retrieval (ISMIR). 2006.

[2] Krebs, Florian, Sebastian Böck, and Gerhard Widmer. "Rhythmic Pattern Modelling for Beat and Downbeat Tracking from Musical Audio." Proc. of the 14th International Conference on Music Information Retrieval (ISMIR), Curitiba. 2013.

License:
--------

Unless indicated otherwise, all source code files are published under the BSD
license and all data/model files under the Creative Commons BY-NC-SA license.
For details, please see the LICENSE file.

If you want to include any of the provided technology in a commercial product,
please contact Gerhard Widmer at gerhard.widmer@jku.at.

Installation:
-------------

To use the much faster viterbi MEX-file, you have to build it first:

1. Within the MATLAB gui, go to the folder `./@HMM`.  
2. Then, execute `mex viterbi.cpp`.  


Contact:
-------------
For comments and bug reports please contact:  
Florian Krebs  
web: http://www.cp.jku.at/people/krebs/  
mail: florian.krebs@jku.at  
