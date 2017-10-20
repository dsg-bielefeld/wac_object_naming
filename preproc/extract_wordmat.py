# coding: utf-8

'''
Compute the feature representations for the image regions

NEVER ever import from here! This is a script that is meant to be run!
When imported, it will run, and potentially overwrite stuff!
(Yes, I know, there should be a main function...)
'''

from __future__ import division

#import json
#import os

import numpy as np
#import pandas as pd

import cPickle as pickle
import gzip
import os
import pandas as pd
import tqdm
import sys
sys.path.append('../TrainModels')
from train_model import create_word2den, wordlist_min_freq,filter_relational_expr

# extracting word matrices for saia and mscoco
# shape: (n words,n samples)
# used by some training scripts for faster sampling


### extracting word matrix for wac vocabulary

def extract_wordmatrix(wlist,refdf,X,oname):
    
    word2den = create_word2den(refdf,wlist)

    print "run word matrix extraction for ", len(wlist), "words"

    Xwords = []
    for i,w in enumerate(wlist):
    #print w
        if i%50 == 0:
            print "."
        
        if w in word2den:
            wvec = [(row[0],row[1],row[2]) in word2den[w] for row in X]
        else:
            print "no denotation found", w
            wvec = [0]*len(X)
        Xwords.append(wvec)
    Xwords = np.array(Xwords)
    print Xwords.shape

    
    np.savez_compressed(oname+'_wmat.npz', Xwords)

    return True


def run_extraction():

    print "load"

    try:
        DATADIRPREF = os.environ['IMAGE_DATA']
    except:
        print "Please provide a bash profile file that specifies the path to your image data"


    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']

    with gzip.open('../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
        refdf_saia = pickle.load(f)

    with gzip.open('../Preproc/PreProcOut/refcoco_refdf.pklz', 'r') as f:
        refdf_coco = pickle.load(f)

    with gzip.open('../Preproc/PreProcOut/grex_refdf.pklz', 'r') as f:
        refdf_grex = pickle.load(f)

    #X_full = np.concatenate([Xsaia, Xcoco])
    refdf_full = pd.concat([refdf_saia, refdf_coco, refdf_grex])

    print "get word list"

    with gzip.open('../TrainModels/TrainedModels/model05_sr5r.pklz', 'r') as f:
        wac_05 = pickle.load(f)
    

    #wlist = wordlist_min_freq(refdf_full,1)
    wlist = wac_05.keys()
    print "Number of words", len(wlist)
    fp = open("wordmats_windex.txt",'w')
    fp.write('\n'.join('%s %s' % x for x in enumerate(wlist)))


    print "get wordmatrix saia"

    extract_wordmatrix(wlist,refdf_saia,Xsaia,'saiapr')


    print "get wordmatrix refcoco"

    extract_wordmatrix(wlist,refdf_coco,Xcoco,'mscoco_refcoco')

    print "get wordmatrix grex"
    extract_wordmatrix(wlist,refdf_grex,Xcoco,'mscoco_grex')

    return True

def run_new_extraction():

    print "load"

    try:
        DATADIRPREF = os.environ['IMAGE_DATA']
    except:
        print "Please provide a bash profile file that specifies the path to your image data"


    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']

    with gzip.open('PreProcOut/saiapr_refdf.pklz', 'r') as f:
        refdf_saia = pickle.load(f)

    with gzip.open('PreProcOut/refcoco_refdf.pklz', 'r') as f:
        refdf_c1 = pickle.load(f)

    with gzip.open('PreProcOut/refcocoplus_refdf.pklz', 'r') as f:
        refdf_c2 = pickle.load(f)

    #X_full = np.concatenate([Xsaia, Xcoco])
    refdf_coco = pd.concat([refdf_c1, refdf_c2])
    refdf_saia = filter_relational_expr(refdf_saia)
    refdf_coco = filter_relational_expr(refdf_coco)

    print "get word list"

    with gzip.open('../TrainModels/TrainedModels/model35_srp5rpos.pklz', 'r') as f:
        wac = pickle.load(f)
    

    #wlist = wordlist_min_freq(refdf_full,1)
    wlist = wac.keys()
    print "Number of words", len(wlist)
    fp = open("wordmats_srp_windex_nr_%d.txt"%len(wlist),'w')
    fp.write('\n'.join(wlist))


    print "get wordmatrix saia"

    extract_wordmatrix(wlist,refdf_saia,Xsaia,'saiapr_nr_%d'%(len(wlist)))


    print "get wordmatrix refcoco"

    extract_wordmatrix(wlist,refdf_coco,Xcoco,'mscoco_refcocoplus_nr_%d'%(len(wlist)))

   

    return True


### extracting word matrix for heads (larger vocabulary)


def extract_headmatrix(headlist,region2heads,X,oname):
    
    print "run word matrix extraction for ", len(headlist), "words"

    Xwords = []
    for i,w in enumerate(headlist):

        if i%100 == 0:
            print ".",
    

        
        wvec = [w in region2heads[(row[0],row[1],row[2])] for row in X]
        Xwords.append(wvec)
    Xwords = np.array(Xwords)
    print Xwords.shape

    np.savez_compressed(oname+'_hmat.npz', Xwords)

    return True

def run_head_extraction():

    print "load"

    try:
        DATADIRPREF = os.environ['IMAGE_DATA']
    except:
        print "Please provide a bash profile file that specifies the path to your image data"


    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']



    with open('../Preproc/PreProcOut/heads_vecs_regions.pklz', 'r') as f:
        (heads2vec,region2head) = pickle.load(f)
    

    #wlist = wordlist_min_freq(refdf_full,1)
    wlist = heads2vec.keys()
    print "Number of words", len(wlist)
    fp = open("headmats_windex.txt",'w')
    fp.write('\n'.join('%s %s' % x for x in enumerate(wlist)))


    print "get wordmatrix saia"

    extract_headmatrix(wlist,region2head,Xsaia,'saiapr')

    print "get wordmatrix mscoco"

    extract_headmatrix(wlist,region2head,Xcoco,'mscoco')

    return True
    
    


### ok, run it!

if __name__ == '__main__':

    #run_extraction()

    #run_head_extraction()

    run_new_extraction()
    



