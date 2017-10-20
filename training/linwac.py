from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json
import os,sys
import random
import yaml
from tqdm import tqdm
from collections import Counter
import sys
from sklearn import linear_model
from scipy.spatial.distance import pdist, squareform
from gensim.models import word2vec


from utils import filter_by_filelist, filter_X_by_filelist
from utils import STOPWORDS, is_relational
from msimilarity import MSimilarity







with gzip.open('../indata/multi_similarity2.pklz', 'r') as f:
    msim = pickle.load(f)

noun_list = [l.strip() for l in open('../indata/noun_list.txt').readlines()]
noun_ind = [msim.word2ind[n] for n in noun_list]




def load_saia_train():

    with open('../indata/saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)


    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    Xsaia_train = Xsaia[np.in1d(Xsaia[:,1],ssplit90['train'])]


    print "Train files", len(ssplit90['train'])
    print "Xsaia_train", Xsaia_train.shape


    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape


    Wsaia_t_train = Wsaia_t[np.in1d(Xsaia[:,1],ssplit90['train'])]
    Wsaia_train = Wsaia_t_train.transpose()

    return Xsaia_train,Wsaia_train

def load_refcoco_train():

    with open('../indata/refcoco_splits.json', 'r') as f:
        rcocosplits = json.load(f)


    X = np.load('../indata/mscoco.npz')
    Xcoco = X['arr_0']
    Xcoco_train = Xcoco[np.in1d(Xcoco[:,1],rcocosplits['train'])]

    print "Xcoco", Xcoco.shape

    W = np.load('../indata/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']

    print "Wcoco", Wcoco.shape

    Wcoco_t = Wcoco.transpose()
    #Wcoco_t.shape


    Wcoco_t_train = Wcoco_t[np.in1d(Xcoco[:,1],rcocosplits['train'])]
    Wcoco_train = Wcoco_t_train.transpose()

    return Xcoco_train,Wcoco_train


def load_w2v():

    

    w2v_z =  (np.array(msim.w2v) - np.nanmean(msim.w2v))/np.nanstd(msim.w2v)
    print w2v_z.shape

    return w2v_z

def load_w2v_rescaled(scale_mean,scale_std):

    

    w2v_z =  (np.array(msim.w2v) - scale_mean)/scale_std
    print w2v_z.shape

    return w2v_z


def load_w2v_top10():

    

    w2v_z =  (np.array(msim.w2v) - np.nanmean(msim.w2v))/np.nanstd(msim.w2v)
    print w2v_z.shape

    w2v_min = np.nanmin(w2v_z)
    w2v_max = np.nanmax(w2v_z)

    w2v_top10 = []

    for wx in range(w2v_z.shape[0]):

        wx_sim =  [w2v_max]*w2v_z.shape[0]
        #print np.argsort(w2v_z[wx])
        top10_x = np.argsort(w2v_z[wx])[:11]
        #print msim.wordlist[wx],[msim.wordlist[tx] for tx in top10_x] 

        for tx in top10_x:
            wx_sim[tx] = w2v_z[wx][tx]
        w2v_top10.append(wx_sim)

    w2v_top10 = np.array(w2v_top10)
    print w2v_top10.shape

    return w2v_top10


def load_w2v_belowmean():

    

    w2v_z =  (np.array(msim.w2v) - np.nanmean(msim.w2v))/np.nanstd(msim.w2v)
    print w2v_z.shape

    w2v_min = np.nanmin(w2v_z)
    w2v_max = np.nanmax(w2v_z)
    w2v_mean = np.nanmean(w2v_z)

    print "Mean sim",w2v_mean

    w2v_top = []

    for wx in range(w2v_z.shape[0]):

        wx_sim =  [w2v_max]*w2v_z.shape[0]
        #print np.argsort(w2v_z[wx])
        top_x = np.where(w2v_z[wx] < -1)[0] # 0 should be mean similarity after z-scoring
        #print top_x
        print msim.wordlist[wx],[msim.wordlist[tx] for tx in top_x] 
        print "*********"

        for tx in top_x:
            wx_sim[tx] = w2v_z[wx][tx]
        w2v_top.append(wx_sim)

    w2v_top = np.array(w2v_top)
    print w2v_top.shape

    return w2v_top



def load_w2v_ref():

    w2v_ref = word2vec.Word2Vec.load('w2v_trained_on_refdf_300dim.mod')


    sim_mat = []
    for x1 in range(len(msim.wordlist)):
        simvec = [w2v_ref.similarity(msim.wordlist[x1],msim.wordlist[x2]) for x2 in range(len(msim.wordlist))]
        sim_mat.append(simvec)

    sim_mat_z = (np.array(sim_mat) - np.nanmean(sim_mat))/np.nanstd(sim_mat)

    return sim_mat_z


def load_trainedsim():

    

    w2v_z =  (np.array(msim.w2v) - np.nanmean(msim.w2v))/np.nanstd(msim.w2v)
    print w2v_z.shape

    w2v_min = np.nanmin(w2v_z)
    w2v_max = np.nanmax(w2v_z)

    bootsim = {}
    for filename in os.listdir('../linmodels/TrainedSimWac'):
    
        if filename.endswith(".pklz"):
            word = filename.split('_')[1]
            wx = msim.word2ind[word]
            print word
            with gzip.open('../linmodels/TrainedSimWac/'+filename, 'r') as f:
                trained = pickle.load(f)
            bootsim[wx] = trained[1]



    trainsim = []
    for wx in range(w2v_z.shape[0]):

        wx_sim =  [w2v_max]*w2v_z.shape[0]
        #print np.argsort(w2v_z[wx])
        if wx in bootsim:
            for tx in bootsim[wx]:
                wx_sim[tx] = bootsim[wx][tx]
            print msim.wordlist[wx],[msim.wordlist[tx] for tx in bootsim[wx] if  bootsim[wx][tx] < 3] 

        trainsim.append(wx_sim)

    trainsim = np.array(trainsim)
    print trainsim.shape

    return trainsim




def extract_sim_nouns(noun,sim_mat,W_train,scaled=False,word_ind=noun_ind):

    wx = msim.word2ind[noun]
    wvec = []

    for rx in range(len(W_train[wx])):
        this_words = np.nonzero(W_train[:,rx])[0]
        this_nouns = [x for x in this_words if x in word_ind]
        
        if len(this_nouns) > 0:
            this_sim = [sim_mat[wx][w2x] for w2x in this_nouns]
            # if np.isnan(this_sim).any():
            #     print this_nouns
            #     print this_sim
            #     print "Nan found", np.min(this_sim)
            wvec.append(np.nanmin(this_sim))

        else:
            #print "nan value", rx, wx, np.nonzero(Wsaia[:,rx])[0]
            #print "First nan", wvec
            wvec.append(np.nan)

    if scaled:
        wvec =  (np.array(wvec) - np.nanmean(wvec))/np.nanstd(wvec)

    return np.array(wvec)


def extract_sim_labels(word,sim_mat,W_train):

    wx = msim.word2ind[word]
    wvec = []

    for rx in range(len(W_train[wx])):
        this_words = np.nonzero(W_train[:,rx])[0]
        #this_words = [x for x in this_words if x in noun_ind]
        
        this_sim = []
        if len(this_words) > 0:
            this_sim = [sim_mat[wx][w2x] for w2x in this_words if not sim_mat[wx][w2x] == np.nan]


        if len(this_sim) > 0:
            wvec.append(np.min(this_sim))
        else:
            wvec.append(np.nan)
    return np.array(wvec)

def extract_sim_ind_labels(word,samplex,sim_mat,W_train):

    wx = msim.word2ind[word]
    labelvec = []

    for rx in samplex:
        this_words = np.nonzero(W_train[:,rx])[0]
        this_sim = [sim_mat[wx][w2x] for w2x in this_words]

        if len(this_sim) > 0:
            labelvec.append(np.nanmin(this_sim))
        else:
            labelvec.append(0.001)

    labelvec = np.array(labelvec)
    labelvec[np.isnan(labelvec)] = 0.001
        
    return np.array(labelvec)

def extract_simdict_ind_labels(word,samplex,sim_dict,W_train):

    wx = msim.word2ind[word]
    wvec = []

    for rx in samplex:
        this_words = np.nonzero(W_train[:,rx])[0]
        #this_words = [x for x in this_words if x in noun_ind]
        
        this_sim = []
        if len(this_words) > 0:
            this_sim = [sim_mat[wx][w2x] for w2x in this_words if not sim_mat[wx][w2x] == np.nan]


        if len(this_sim) > 0:
            wvec.append(np.min(this_sim))
        else:
            wvec.append(np.nan)
    return np.array(wvec)

    #wsim_mat.append(wvec)

def train_noun(noun,sim_mat,W_train,X_train,scaled=False,word_ind=noun_ind):

    y = extract_sim_nouns(noun,sim_mat,W_train,scaled=scaled,word_ind=word_ind)
    #print len(y)
    #print len(y[np.isnan(y)])

    this_X = X_train[~np.isnan(y)]
    this_X = this_X[:,3:]
    this_y = y[~np.isnan(y)]

    reg = None

    print "this_X", this_X.shape

    if len(this_X) > 0:
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_X,this_y)
    else:
        print "Empty classifier", noun

    return reg

def train_word(word,sim_mat,W_train,X_train):

    y = extract_sim_labels(word,sim_mat,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

    this_X = X_train[~np.isnan(y)]
    this_X = this_X[:,3:]
    this_y = y[~np.isnan(y)]

    try: 
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_X,this_y)
    except:
        reg = None
        print "Could not train", word, len(this_y), len(this_X)

    return reg


def simsample_word(word,sim_mat,W_train,X_train,nsim=5,nneg=10):

    wx = msim.word2ind[word]
    pos_ind = W_train[wx].nonzero()[0]

    sim_wx = np.argsort(sim_mat[wx])[1:nsim+1]
        #print sim_wx
    sim_ind = []
    for sim_w in sim_wx:
        sim_ind += list(W_train[sim_w].nonzero()[0])
        
    sim_ind = np.array(list(set(sim_ind)))

    neg_ind = [i for i in range(W_train[wx].shape[0]) if not (i in pos_ind or i in sim_ind)]
    if len(neg_ind) > (len(pos_ind)+len(sim_ind))*nneg:
        neg_ind = random.sample(neg_ind,(len(pos_ind)+len(sim_ind))*nneg)

    if len(neg_ind) > 0:
        neg_ind = np.array(neg_ind)
        neg_X = X_train[neg_ind]
        neg_y = extract_sim_ind_labels(word,neg_ind,sim_mat,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

        neg_X = neg_X[~np.isnan(neg_y)]
        neg_X = neg_X[:,3:]
        neg_y = neg_y[~np.isnan(neg_y)]

        pos_X = X_train[pos_ind]
        pos_X = pos_X[:,3:]
        pos_y = np.array([np.nanmin(sim_mat)]*len(pos_X))

        if len(sim_ind) > 0:
            sim_X = X_train[sim_ind]
            sim_y = extract_sim_ind_labels(word,sim_ind,sim_mat,W_train)

            sim_X = sim_X[~np.isnan(sim_y)]
            sim_X = sim_X[:,3:]
            sim_y = sim_y[~np.isnan(sim_y)]

            this_X = np.vstack([pos_X,sim_X,neg_X])
            this_y = np.hstack([pos_y,sim_y,neg_y])

        else:
            this_X = np.vstack([pos_X,neg_X])
            this_y = np.hstack([pos_y,neg_y])

    else:
        print "Word", word, "no neg samples"
        this_X = []
        this_y = []

    return this_X,this_y


def simsample_word_prop(word,sim_mat,W_train,X_train,nsim=5,nneg=0.1):

    wx = msim.word2ind[word]
    pos_ind = W_train[wx].nonzero()[0]

    sim_ind = []
    if nsim > 0:
        sim_wx = np.argsort(sim_mat[wx])[1:nsim+1]
        #print sim_wx
        for sim_w in sim_wx:
            sim_ind += list(W_train[sim_w].nonzero()[0])    
    sim_ind = np.array(list(set(sim_ind)))

    neg_sample = int(len(W_train[wx])*nneg)
    neg_ind = [i for i in range(W_train[wx].shape[0]) if not (i in pos_ind or i in sim_ind)]
    if len(neg_ind) > neg_sample:
        neg_ind = random.sample(neg_ind,neg_sample)

    if len(neg_ind) > 0:
        neg_ind = np.array(neg_ind)
        neg_X = X_train[neg_ind]
        neg_y = extract_sim_ind_labels(word,neg_ind,sim_mat,W_train)
    #print len(y)
    #print len(y[np.isnan(y)])

        neg_X = neg_X[~np.isnan(neg_y)]
        neg_X = neg_X[:,3:]
        neg_y = neg_y[~np.isnan(neg_y)]

        pos_X = X_train[pos_ind]
        pos_X = pos_X[:,3:]
        pos_y = np.array([np.nanmin(sim_mat)]*len(pos_X))

        if len(sim_ind) > 0:
            sim_X = X_train[sim_ind]
            sim_y = extract_sim_ind_labels(word,sim_ind,sim_mat,W_train)

            sim_X = sim_X[~np.isnan(sim_y)]
            sim_X = sim_X[:,3:]
            sim_y = sim_y[~np.isnan(sim_y)]

            this_X = np.vstack([pos_X,sim_X,neg_X])
            this_y = np.hstack([pos_y,sim_y,neg_y])

        else:
            this_X = np.vstack([pos_X,neg_X])
            this_y = np.hstack([pos_y,neg_y])

    else:
        print "Word", word, "no neg samples"
        this_X = []
        this_y = []

    return this_X,this_y


def abssample_word(word,sim_mat,W_train,X_train,nsamp=100):

    wx = msim.word2ind[word]
    pos_ind = W_train[wx].nonzero()[0]

    #if len(pos_ind) > 0:

    if len(pos_ind) > nsamp/2:
        pos_ind = np.array(random.sample(list(pos_ind),int(nsamp/2)))


    nsamp_pos = len(pos_ind)
    nsamp_neg = nsamp-nsamp_pos

    neg_ind = np.where(W_train[wx] == 0)[0]
    if len(neg_ind) > nsamp_neg:
        neg_ind = np.array(random.sample(list(neg_ind),nsamp_neg))


    neg_X = X_train[neg_ind]
    neg_y = extract_sim_ind_labels(word,neg_ind,sim_mat,W_train)


    neg_X = neg_X[~np.isnan(neg_y)]
    neg_X = neg_X[:,3:]
    neg_y = neg_y[~np.isnan(neg_y)]

    pos_X = X_train[pos_ind]
    pos_X = pos_X[:,3:]
    pos_y = np.array([np.nanmin(sim_mat)]*len(pos_X))

    this_X = np.vstack([pos_X,neg_X])
    this_y = np.hstack([pos_y,neg_y])

    #else:
    #    print "Word", word, "no positive samples"
    #    this_X = []
    #    this_y = []

    return this_X,this_y



def train_ridge(this_X,this_y):
    try: 
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(this_X,this_y)
    except:
        print "Could not train"
        reg = None

    return reg

def train_all_nouns(W_train,X_train,sim_mat,ssim="w2v",scaled=False,word_list=noun_list):

    this_ind = [msim.word2ind[w] for w in word_list]

    print "Linwac based on distributional similarity"
    linwac_w2v = {}
    for some_n in tqdm(word_list):
        linwac_w2v[some_n] = train_noun(some_n,sim_mat,W_train,X_train,scaled=True,word_ind=this_ind)

    with gzip.open('../linmodels/linwac_nouns_'+ssim+'.pklz', 'w') as f:
        pickle.dump(linwac_w2v, f)

    # print "Onto wac based on visual similarity"
    # linwac_visual = {}
    # for some_n in tqdm(noun_list):
    #     linwac_visual[some_n] = train_noun(some_n,visual_z)

    # with gzip.open('linwac_nouns_visual.pklz', 'w') as f:
    #     pickle.dump(linwac_visual, f)





def train_saia_propsample(W_train,X_train,sim_mat,nsim=5,nneg=0.1,ssim="w2v"):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    linwac_w2v = {}
    for some_n in tqdm(word_list):
        linwac_w2v[some_n] = train_ridge(*simsample_word_prop(some_n,sim_mat,W_train,X_train,nsim=nsim,nneg=nneg))

    with gzip.open('../linmodels/linwac_vocab_'+ssim+'_nsim'+ str(nsim) + '_nneg'+str(nneg)+'.pklz', 'w') as f:
        pickle.dump(linwac_w2v, f)


def train_vocab_abssample(W_train,X_train,sim_mat,nsamp=100,ssim="w2v",scorp="saia",word_list=msim.wordlist):

    #word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    linwac_w2v = {}
    for some_n in tqdm(word_list):
        linwac_w2v[some_n] = train_ridge(*abssample_word(some_n,sim_mat,W_train,X_train,nsamp=nsamp))

    with gzip.open('../linmodels/linwac_vocab_%s_%s_nsamp%s.pklz'%(scorp,ssim,str(nsamp)), 'w') as f:
        pickle.dump(linwac_w2v, f)

def train_combined((X_train1,W_train1),(X_train2,W_train2),sim_mat,nsim=5,nneg=10,ssim="w2v"):

    word_list = msim.word2ind.keys()
    print "words",len(word_list)

    print "Onto wac based on distributional similarity"
    linwac_w2v = {}
    for some_n in tqdm(word_list):
        X1,y1 = simsample_word(some_n,sim_mat,W_train1,X_train1,nsim=nsim,nneg=nneg)
        X2,y2 = simsample_word(some_n,sim_mat,W_train2,X_train2,nsim=nsim,nneg=nneg)
        linwac_w2v[some_n] = train_ridge(np.vstack([X1,X2]),np.hstack([y1,y2]))

    with gzip.open('../linmodels/linwac_saiarefc_' + ssim + '_nsim'+ str(nsim) + '_nneg'+str(nneg)+'.pklz', 'w') as f:
        pickle.dump(linwac_w2v, f)




if __name__ == '__main__':

    

    w2v = load_w2v()
    saia_t = load_saia_train()
    train_vocab_abssample(saia_t[1],saia_t[0],w2v,nsamp=74390,ssim="w2v")


    