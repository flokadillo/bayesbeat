*bayesbeat* is a MATLAB package for metrical structure analysis of mucial audio signals. It is based on the *dynamic bar pointer model* first proposed in [1]. The model was later extended by various authors. 

It includes algorithms for inference with *Hidden Markov Models* (HMMs) and *Particle Filter* models (PF). For detailed information about the algorithms, please also see the References section.

License
=======

The package has two licenses, one for source code and one for model/data files.

Source code
-----------

Unless indicated otherwise, all source code files are published under the BSD
license. For details, please see the [LICENSE](./LICENSE) file.

Model and data files
--------------------

Unless indicated otherwise, all model and data files are distributed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.

If you want to include any of these files (or a variation or modification
thereof) or technology which utilises them in a commercial product, please
contact [Gerhard Widmer](http://www.cp.jku.at/people/widmer/).

Installation:
=============

To use the much faster viterbi MEX-file, you have to build it first:

1. Within the MATLAB gui, go to the folder [src/@HiddenMarkovModel](./src/@HiddenMarkovModel).  
2. Then, execute `mex viterbi_cpp.cpp`.  

Package structure
=================

The package has a very simple structure, divided into the following folders:

* [/apps](./apps) this folder includes applications (i.e. executable algorithms)  
* [/doc](./doc) documentation  
* [/examples](./examples) examples of how to use the package  
* [/scr](./src) source code of the package  
* [/tests](./tests) tests  

Getting started
===============
The bayesbeat package is best explored by looking at the [examples](./examples) folder. E.g., you can try the following examples:

* Example 1: Compute beats using a pretrained HMM model  
* Example 2: Compute beats using a pretrained PF model  
* Example 3/4: Learn the HMM observation model parameters from training data and set up a HMM  
* Example 5: Learn the HMM observation model parameters from training data and set up a PF model

Contact:
=======
For comments and bug reports please contact:  
[Florian Krebs](http://www.cp.jku.at/people/krebs)  

References
==========

[1] Whiteley, N., Cemgil A. T., and Godsill S.. *Bayesian Modelling of Temporal Structure in Musical Audio.* Proceedings of the 14th International Conference on Music Information Retrieval (ISMIR). 2006.

[2] Whiteley, N., Cemgil, A. T., Godsill, S.. *Sequential inference of rhythmic structure in musical audio.* Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2007.

[3] Krebs, F., Böck S., and Widmer G.. *Rhythmic Pattern Modelling for Beat and Downbeat Tracking from Musical Audio.* Proceedings of the 14th International Conference on Music Information Retrieval (ISMIR), Curitiba. 2013.

[4] Krebs, F., Holzapfel, A., Cemgil, A. T., and Widmer, G.. *Inferring Metrical Structure in Music Using Particle Filters.* In IEEE/ACM Transactions on Audio, Speech, and Language Processing. 2015.

[5] Holzapfel, A., Krebs, F., Srinivasamurthy, A.. *Tracking the "odd": Meter inference in a culturally diverse music corpus.* Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014.

[6] Srinivasamurthy, A., Holzapfel, A., Cemgil, A., Serra, X.. *Particle Filters for Efficient Meter Tracking with Dynamic Bayesian networks*. Proceedings of the 16th International Society for Music Information Retrieval Conference (ISMIR), 2015.

[7] Krebs, F., Böck, S., and Widmer, G.. *An Efficient State Space Model for Joint Tempo and Meter Tracking.* In Proceedings of 16th International Society for Music Information Retrieval Conference (ISMIR), Malaga, Spain, 2015.

[8] Böck, S., Krebs, F., and Schedl., M.. *Evaluating the Online Capabilities of Onset Detection Methods.* In Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), Porto, Portugal, 2012. 

