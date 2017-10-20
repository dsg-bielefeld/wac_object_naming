from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json
import random
import yaml
from tqdm import tqdm
from collections import Counter

from utils import filter_by_filelist, filter_X_by_filelist
from scipy.spatial.distance import pdist, squareform
from msimilarity import MSimilarity
from linwac import load_saia_train
from sklearn import linear_model






def extract_sim_nouns(sim_vecs,noun_ind,W_train):

    y_mat = []
    y_shape = len(sim_vecs[0])
    print "Y-Shape",y_shape

    for rx in range(W_train.shape[1]):
    #for rx in range(1000):
        #print rx
        this_words = np.nonzero(W_train[:,rx])[0]
        #print this_words
        this_nouns = [x for x in this_words if x in noun_ind]
        #print this_nouns


        if len(this_nouns) > 0:
            y_mat.append(np.array(sim_vecs[this_nouns[0]]).astype(np.float))

        else:
            #print "nan value", rx, wx, np.nonzero(Wsaia[:,rx])[0]
            #print "First nan", wvec
            y_mat.append(np.array([np.nan]*y_shape).astype(float))

    y_mat = np.array(y_mat)
    print "final y mat shape", y_mat.shape
    print "final y mat without nans shape", y_mat[~np.isnan(y_mat).any(axis=1)].shape
    return y_mat

def extract_composed_vecs(sim_vecs,W_train):

    y_mat = []
    y_shape = len(sim_vecs[0])
    print "Y-Shape",y_shape

    for rx in range(W_train.shape[1]):
        this_words = np.nonzero(W_train[:,rx])[0]
        
        if len(this_words) > 0:
            y_vecs = np.array([np.array(sim_vecs[wx]).astype(np.float) for wx in this_words])
            y_mat.append(np.mean(y_vecs,axis=0)) 

        else:
            #print "nan value", rx, wx, np.nonzero(Wsaia[:,rx])[0]
            #print "First nan", wvec
            y_mat.append(np.array([np.nan]*y_shape).astype(float))

    y_mat = np.array(y_mat)
    print "final y mat shape", y_mat.shape
    return y_mat

def train_mappings(sim_mat,noun_ind,W_train,X_train,split=""):

    y = extract_sim_nouns(sim_mat,noun_ind,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

    #print y

    this_X = X_train[~np.isnan(y).any(axis=1)]
    this_X = this_X[:,3:]
    this_y = y[~np.isnan(y).any(axis=1)]



    maplin = []
    for dx in tqdm(range(this_y.shape[1])):   
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_X,this_y[:,dx])
        maplin.append(reg)        


    with gzip.open('../linmodels/linmap_nouns'+split+'.pklz', 'w') as f:
        pickle.dump(maplin, f)

    return maplin

def train_composed_mappings(sim_mat,W_train,X_train):

    y = extract_composed_vecs(sim_mat,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

    #print y

    this_X = X_train[~np.isnan(y).any(axis=1)]
    this_X = this_X[:,3:]
    this_y = y[~np.isnan(y).any(axis=1)]



    maplin = []
    for dx in range(this_y.shape[1]):   
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_X,this_y[:,dx])
        maplin.append(reg)        


    with gzip.open('../linmodels/linmap_composed.pklz', 'w') as f:
        pickle.dump(maplin, f)

    return maplin

def train_composed_mappings_text2image(sim_mat,W_train,X_train):

    y = extract_composed_vecs(sim_mat,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

    #print y

    this_X = X_train[~np.isnan(y).any(axis=1)]
    this_X = this_X[:,3:]
    this_y = y[~np.isnan(y).any(axis=1)]



    maplin = []
    for dx in range(this_X.shape[1]):   
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_y,this_X[:,dx])
        maplin.append(reg)        


    with gzip.open('../linmodels/linmap_composed_text2image.pklz', 'w') as f:
        pickle.dump(maplin, f)

    return maplin

if __name__ == '__main__':

    with gzip.open('../indata/multi_similarity2.pklz', 'r') as f:
        msim = pickle.load(f)

    noun_list = [l.strip() for l in open('../indata/noun_list.txt').readlines()]
    print "Nouns", len(noun_list)
    noun_ind = [msim.word2ind[n] for n in noun_list] 
    

    (Xsaia_t,Wsaia_t) = load_saia_train()
    extract_sim_nouns(msim.w2v_vecs,noun_ind,Wsaia_t)
    train_mappings(msim.w2v_vecs,noun_ind,Wsaia_t,Xsaia_t)
    #train_composed_mappings_text2image(msim.w2v_vecs,Wsaia_t,Xsaia_t)


