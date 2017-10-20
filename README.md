# Object Naming

This repository contains the code required to reproduce the results reported in

* Sina Zarrieß, David Schlangen. 2017. Obtaining referential word meanings from visual and distributional information: Experiments on object naming. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2017)*, Vancouver, Canada.*


The work reported here is based on image and referring expression corpora collected by other groups (the images themselves are from the "IAPR TC-12" dataset ( <http://www.imageclef.org/photodata>), the referring expressions have been collected using the [ReferIt game](http://tamaraberg.com/referitgame/) by Tamara Berg and colleagues). 

The work is also based on procedures for data preprocessing used for other papers, see our repository on the [words-as-classifiers model](https://github.com/dsg-bielefeld/image_wac). For convenience, the various files obtained from preprocessing including pre-trained models can be found [here](https://uni-bielefeld.sciebo.de/index.php/s/owCGZOtSQXeXoVp).

The structure of the folder should look like this:

├── README.md
├── evaluation
│   ├── apply_model_matrix.py
│   ├── eval_gen_saia.py
│   └── eval_zeroshot_saia.py
├── indata
│   ├── multi_similarity2.pklz 
│   ├── noun_list.txt
│   ├── noun_list_long.txt
│   ├── saia_nouns159_testset.pklz 
│   ├── saia_zeroshot_hypernsplit.json
│   ├── saia_zeroshot_hypernsplit_testset.pklz 
│   ├── saia_zeroshot_mixedpluralsplit.json
│   ├── saia_zeroshot_mixedpluralsplit_testset.pklz
│   ├── saia_zeroshot_nounslong_splits.json
│   ├── saia_zeroshot_nounsplits.json
│   ├── saia_zeroshot_nounsplits_testsets.json
│   ├── saia_zeroshot_nounsplits_testsets.pklz
│   ├── saia_zeroshot_pluralsplit.json 
│   ├── saia_zeroshot_pluralsplit_testset.pklz
│   ├── saiapr.npz 
│   ├── saiapr_90-10_splits.json
│   ├── saiapr_refdf.pklz
│   ├── saiapr_windex.txt 
│   └── saiapr_wmat.npz
├── linmodels
│   ├── linmap_nouns.pklz
│   ├── linmap_nouns_zeroshot_hypernsplit.pklz
│   ├── linmap_nouns_zeroshot_mixedpluralsplit.pklz
│   ├── linmap_nouns_zeroshot_pluralsplit.pklz
│   ├── linmap_nouns_zeroshot_split0.pklz
│   ├── linmap_nouns_zeroshot_split1.pklz
│   ├── linmap_nouns_zeroshot_split10.pklz
│   ├── linmap_nouns_zeroshot_split2.pklz
│   ├── linmap_nouns_zeroshot_split3.pklz
│   ├── linmap_nouns_zeroshot_split4.pklz
│   ├── linmap_nouns_zeroshot_split5.pklz
│   ├── linmap_nouns_zeroshot_split6.pklz
│   ├── linmap_nouns_zeroshot_split7.pklz
│   ├── linmap_nouns_zeroshot_split8.pklz
│   ├── linmap_nouns_zeroshot_split9.pklz
│   ├── linwac_nouns__standard_pluralsplit.pklz
│   ├── linwac_nouns__zeroshot_hypernsplit.pklz
│   ├── linwac_nouns__zeroshot_mixedpluralsplit.pklz
│   ├── linwac_nouns__zeroshot_pluralsplit.pklz
│   ├── linwac_nouns_w2v_zeroshot_split0.pklz
│   ├── linwac_nouns_w2v_zeroshot_split1.pklz
│   ├── linwac_nouns_w2v_zeroshot_split10.pklz
│   ├── linwac_nouns_w2v_zeroshot_split2.pklz
│   ├── linwac_nouns_w2v_zeroshot_split3.pklz
│   ├── linwac_nouns_w2v_zeroshot_split4.pklz
│   ├── linwac_nouns_w2v_zeroshot_split5.pklz
│   ├── linwac_nouns_w2v_zeroshot_split6.pklz
│   ├── linwac_nouns_w2v_zeroshot_split7.pklz
│   ├── linwac_nouns_w2v_zeroshot_split8.pklz
│   ├── linwac_nouns_w2v_zeroshot_split9.pklz
│   └── linwac_vocab_w2v_nsamp74390.pklz
├── logmodels
│   ├── logwac_saia_nouns_standard_pluralsplit_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_hypernsplit_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_pluralsplit_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split0_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split1_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split2_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split3_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split4_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split5_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split6_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split7_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split8_nosamp.pklz
│   ├── logwac_saia_nouns_zeroshot_split9_nosamp.pklz
│   └── logwac_saia_w2v_nsamp74390.pklz
├── preproc
│   ├── extract_wordmat.py
│   └── refexp_heads.py
└── training
    ├── linmap.py
    ├── linwac.py
    ├── logwac.py
    ├── msimilarity.py
    ├── utils.py
    └── zero_shot_models.py


You can do two things now:

1) retrain your own models, see training/
    * linmap.py: train transfer models
    * linwac.py: train words-as-classifiers with distributional supervision signal
    * logwac.py: train standard words-as-classifiers (with a lot of negative samples)
    * zero_shot_models.py: train models for zero-shot object naming

2) evaluate the models, see models/

*Sina Zarrieß, 2017-10-20* 
