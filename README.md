# Object Naming

This repository contains the code required to reproduce the results reported in

* Sina Zarrieß, David Schlangen. 2017. Obtaining referential word meanings from visual and distributional information: Experiments on object naming. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2017)*, Vancouver, Canada.*


The work reported here is based on image and referring expression corpora collected by other groups (the images themselves are from the "IAPR TC-12" dataset ( <http://www.imageclef.org/photodata>), the referring expressions have been collected using the [ReferIt game](http://tamaraberg.com/referitgame/) by Tamara Berg and colleagues). 

The work is also based on procedures for data preprocessing used for other papers, see our repository on the [words-as-classifiers model](https://github.com/dsg-bielefeld/image_wac). For convenience, the various files obtained from preprocessing including pre-trained models can be found [here](https://uni-bielefeld.sciebo.de/index.php/s/owCGZOtSQXeXoVp).

The structure of the folder should look like this:
```
├── evaluation
├── indata
├── linmodels
├── logmodels
├── preproc
└── training
```

You can do two things now:

1) retrain your own models, see training/
    * linmap.py: train transfer models
    * linwac.py: train words-as-classifiers with distributional supervision signal
    * logwac.py: train standard words-as-classifiers (with a lot of negative samples)
    * zero_shot_models.py: train models for zero-shot object naming

2) evaluate the models, see models/

*Sina Zarrieß, 2017-10-20* 
