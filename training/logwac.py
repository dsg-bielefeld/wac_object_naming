from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json
import yaml
from tqdm import tqdm
import scipy.stats
from collections import Counter
import random
from scipy.spatial.distance import pdist, squareform


from msimilarity import MSimilarity
from sklearn import linear_model
from linwac import load_saia_train, load_refcoco_train
from utils import filter_by_filelist, filter_X_by_filelist
from utils import STOPWORDS, is_relational


with gzip.open('../indata/multi_similarity2.pklz', 'r') as f:
    msim = pickle.load(f)

noun_list = [l.strip() for l in open('../indata/noun_list.txt').readlines()]
noun_ind = [msim.word2ind[n] for n in noun_list]

w2v_z =  (np.array(msim.w2v) - np.nanmean(msim.w2v))/np.nanstd(msim.w2v)
print w2v_z.shape

visual_z =  (np.array(msim.visual) - np.nanmean(msim.visual))/np.nanstd(msim.visual)
print visual_z.shape

def simsample_word(word,sim_mat,W_train,X_train,nsim=5):

    wx = msim.word2ind[word]
    pos_ind = W_train[wx].nonzero()[0]

    if len(pos_ind) > 0:

        
        neg_ind = set(np.where(W_train[wx] == 0)[0])

        sim_wx = np.argsort(sim_mat[wx])[1:nsim+1]
        sim_ind = []
        for this_sim_wx in sim_wx:
            sim_ind += list(np.where(W_train[this_sim_wx] == 0)[0])
        if len(sim_ind) > (len(neg_ind)*0.5):
            sim_ind = sim_ind[:int(len(neg_ind)*0.5)]

        neg_ind = list(neg_ind - set(sim_ind))

        neg_X = X_train[neg_ind]
        neg_X = neg_X[:,3:]
        neg_y = np.array([0]*neg_X.shape[0])


        pos_X = X_train[pos_ind]
        pos_X = pos_X[:,3:]
        pos_y = np.array([1]*pos_X.shape[0])


        this_X = np.vstack([pos_X,neg_X])
        this_y = np.hstack([pos_y,neg_y])

    else:

        this_X = []
        this_y = []


    return this_X,this_y

def abssample_word(word,sim_mat,W_train,X_train,nsamp=100):

    wx = msim.word2ind[word]
    pos_ind = W_train[wx].nonzero()[0]

    if len(pos_ind) > 0:

        if len(pos_ind) > nsamp/2:
            pos_ind = np.array(random.sample(list(pos_ind),int(nsamp/2)))

        nsamp_pos = len(pos_ind)
        nsamp_neg = nsamp-nsamp_pos

        neg_ind = np.where(W_train[wx] == 0)[0]
        if len(neg_ind) > nsamp_neg:
            neg_ind = np.array(random.sample(list(neg_ind),nsamp_neg))

        neg_X = X_train[neg_ind]
        neg_X = neg_X[:,3:]
        neg_y = np.array([0]*neg_X.shape[0])


        pos_X = X_train[pos_ind]
        pos_X = pos_X[:,3:]
        pos_y = np.array([1]*pos_X.shape[0])


        this_X = np.vstack([pos_X,neg_X])
        this_y = np.hstack([pos_y,neg_y])

    else:

        this_X = []
        this_y = []

    

    return this_X,this_y

def train_logreg(this_X,this_y):
    try: 
        reg = linear_model.LogisticRegression(penalty='l1')
        reg.fit(this_X,this_y)
    except:
        print "Could not train"
        reg = None

    return reg

def train_combined((X_train1,W_train1),(X_train2,W_train2),nsim=5,nrand=10):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    logwac_w2v = {}
    for some_n in tqdm(word_list):
        X1,y1 = simsample_word(some_n,w2v_z,W_train1,X_train1,nsim=nsim,nrand=nrand)
        X2,y2 = simsample_word(some_n,w2v_z,W_train2,X_train2,nsim=nsim,nrand=nrand)
        logwac_w2v[some_n] = train_logreg(np.vstack([X1,X2]),np.hstack([y1,y2]))

    with gzip.open('../logmodels/logwac_saiarefc_w2v_nsim'+ str(nsim) + '_nrand'+str(nrand)+'.pklz', 'w') as f:
        pickle.dump(logwac_w2v, f)

def train_saia(X_train,W_train,nneg=5):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    logwac_w2v = {}
    for some_n in tqdm(word_list):
        X1,y1 = simsample_word(some_n,w2v_z,W_train,X_train,nneg=nneg)
        #X2,y2 = simsample_word(some_n,w2v_z,W_train2,X_train2,nsim=nsim,nrand=nrand)
        logwac_w2v[some_n] = train_logreg(X1,y1)

    with gzip.open('../logmodels/logwac_saia_w2v_nneg'+ str(nneg) +'.pklz', 'w') as f:
        pickle.dump(logwac_w2v, f)

def train_saia_abssamp(X_train,W_train,nsamp=100):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    logwac_w2v = {}
    for some_n in tqdm(word_list):
        X1,y1 = abssample_word(some_n,w2v_z,W_train,X_train,nsamp=nsamp)
        #X2,y2 = simsample_word(some_n,w2v_z,W_train2,X_train2,nsim=nsim,nrand=nrand)
        logwac_w2v[some_n] = train_logreg(X1,y1)

    with gzip.open('../logmodels/logwac_saia_w2v_nsamp'+ str(nsamp) +'.pklz', 'w') as f:
        pickle.dump(logwac_w2v, f)

def train_saia_simsamp(X_train,W_train,nsim=5):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Wac, removing similar words from negative samples"
    logwac_w2v = {}
    for some_n in tqdm(word_list):
        X1,y1 = simsample_word(some_n,w2v_z,W_train,X_train,nsim=nsim)
        #X2,y2 = simsample_word(some_n,w2v_z,W_train2,X_train2,nsim=nsim,nrand=nrand)
        logwac_w2v[some_n] = train_logreg(X1,y1)
        #logwac_w2v[some_n]['nsamp'] = X1.shape[0]

    with gzip.open('../logmodels/logwac_saia_w2v_remove_nsim'+ str(nsim) +'.pklz', 'w') as f:
        pickle.dump(logwac_w2v, f)

def train_saia_nosamp(X_train,W_train,word_list=None,ssim=""):

    if not word_list:
        word_list = msim.word2ind.keys()

    print "words",len(word_list)

    logwac = {}
    for some_n in tqdm(word_list):

        this_y = W_train[msim.word2ind[some_n]] == 1
        this_X = X_train[:,3:]
        #X1,y1 = abssample_word(some_n,w2v_z,W_train,X_train,nsamp=nsamp)
        #X2,y2 = simsample_word(some_n,w2v_z,W_train2,X_train2,nsim=nsim,nrand=nrand)
        logwac[some_n] = train_logreg(this_X,this_y)

    with gzip.open('../logmodels/logwac_saia_'+ssim+'_nosamp.pklz', 'w') as f:
        pickle.dump(logwac, f)


if __name__ == '__main__':

    saia_t = load_saia_train()
    train_saia_abssamp(saia_t[0],saia_t[1],nsamp=74390)

