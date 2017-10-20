from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json
import os

import scipy.stats
import scipy
from scipy.spatial.distance import pdist, squareform


from tqdm import tqdm
from itertools import combinations
from collections import Counter

import sys
sys.path.append('../../Exploration')

from similarity import set_up_denotational_vectors, set_up_denotational_similarity
# das 2016-09-28: I have to comment out the import of s_u_den_vec and den_sim
#  because otherwise I get an import error!
from similarity import turn_into_distr, pairwise_mi_by_ind
from similarity import turn_into_distr

sys.path.append('../../Utils')
from utils import filter_by_filelist, filter_X_by_filelist

import yaml
from time import strftime
import math

NTEST = 500

try:
    # DATADIRPREF = os.environ['IMAGE_DATA']
    config = yaml.load(file('slackbot.conf', 'r'))
    DATADIRPREF = config['IMAGE_DATA']
except:
    print "Please provide a config file that specifies the path to your image data"

SIMPATH = 'Models/similarities.pklz'
SAIAPR_CATPATH = DATADIRPREF + '/Corpora/Others/ImageCorpora/IAPR_ReferIt/SAIA_Data/benchmark/wlist.txt'

def load_data(stopped=False):

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    print "Xsaia", Xsaia.shape

    L = np.load('./saiapr_vgg19-prd.npz')
    Lsaia = L['arr_0']

    print "Lsaia", Lsaia.shape

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']

    print "Xcoco", Xcoco.shape

    W = np.load('../TrainModels/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    W = np.load('../TrainModels/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']

    print "Wcoco", Wcoco.shape

    #wfile = open('../TrainModels/wordmats_windex.txt','r').readlines()
    #ilist = [tuple(l.split()) for l in wfile]
    #ilist = [(int(a),b) for a,b in ilist]
    #ind2w = dict(ilist)
    with gzip.open('../TrainModels/TrainedModels/model05_sr5r.pklz', 'r') as f:
        wac_05 = pickle.load(f)

    with gzip.open('linwac_vocab_w2v.pklz', 'r') as f:
        linwac = pickle.load(f)

    with open('../Preproc/PreProcOut/refcoco_splits.json', 'r') as f:
        rcocosplits = json.load(f)
    r_testfiles = rcocosplits['testB']

    with open('../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)
    s_testfiles = ssplit90['test']

    s_testfiles = s_testfiles[:-NTEST]
    s_testfiles_den = s_testfiles[-NTEST:]
    r_testfiles_den = r_testfiles[:NTEST]  # could use testA, but have filtered bbdf above..
    filelist_w_icorp = [(0, image_id) for image_id in s_testfiles_den]
    filelist_w_icorp.extend([(1, image_id) for image_id in r_testfiles_den])

    X_full = np.concatenate([Xsaia, Xcoco])

    with gzip.open('../../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
        s_bbdf = pickle.load(f)
    s_bbdf = filter_by_filelist(s_bbdf, s_testfiles)

    with gzip.open('../../Preproc/PreProcOut/mscoco_bbdf.pklz', 'r') as f:
        c_bbdf = pickle.load(f)
    c_bbdf = filter_by_filelist(c_bbdf, r_testfiles)

    bbdf = pd.concat([s_bbdf, c_bbdf])


    with gzip.open('../../Preproc/PreProcOut/refcoco_refdf.pklz', 'r') as f:
        cocrefdf = pickle.load(f)
    with gzip.open('../../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
        srefdf = pickle.load(f)
    refdf = pd.concat([srefdf, cocrefdf])
    #***

    #if stopped:
    #    wlist = [word for word in wac_05.keys() if not word in ['on','to','from','of','at','in','with','us','is','you','what']]
     #   word2ind = dict([(word, no) for no, word in enumerate(wlist)])

    #else:
    word2ind = dict([(word, no) for no, word in enumerate(wac_05.keys())])
    wlist = [word for word in wac_05.keys()]

    print "Set up set_up_denotational_similarity"

    den_mat = set_up_denotational_similarity(X_full, bbdf, filelist_w_icorp,
                                   wac_05, word2ind, nobj=NTEST)
    den_vecs = set_up_denotational_vectors(X_full, bbdf, filelist_w_icorp,
                                   wac_05, nobj=NTEST)

    print "Set up set_up_lindenotational_similarity"

    linden_mat = set_up_lindenotational_similarity(X_full, bbdf, filelist_w_icorp,
                                   linwac, word2ind, nobj=NTEST)

    print "Set up visual similarity"

    vis_vecs,vis_mat = set_up_visual_similarity(wlist,Xsaia,Xcoco,Wsaia,Wcoco)


    print "Set up w2v similarity"

    w2v_vecs,w2v_mat = set_up_w2v_similarity(wlist)

    #lab_mat = set_up_label_similarity(wlist,Xsaia,Wsaia)

    print "Context matrix"

    ctxtfiles = [(1, image_id) for image_id in rcocosplits['train']] + \
    [(0, image_id) for image_id in ssplit90['train']]
    context_pmi, context_sq = set_up_context_associations(refdf,wlist,ctxtfiles)


    sim = MSimilarity((den_mat,den_vecs), (vis_mat,vis_vecs), (w2v_mat,w2v_vecs), (context_sq,context_pmi), linden_mat, word2ind, wlist)

    return sim

def load_linmat(stopped=False):

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    print "Xsaia", Xsaia.shape

    

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']

   
    with gzip.open('../TrainModels/TrainedModels/model05_sr5r.pklz', 'r') as f:
        wac_05 = pickle.load(f)

    with gzip.open('linmodels/linwac_saiarefc_w2v_nsim10_nneg10.pklz', 'r') as f:
        linwac = pickle.load(f)

    with open('../Preproc/PreProcOut/refcoco_splits.json', 'r') as f:
        rcocosplits = json.load(f)
    r_testfiles = rcocosplits['testB']

    with open('../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)
    s_testfiles = ssplit90['test']

    s_testfiles = s_testfiles[:-NTEST]
    s_testfiles_den = s_testfiles[-NTEST:]
    r_testfiles_den = r_testfiles[:NTEST]  # could use testA, but have filtered bbdf above..
    filelist_w_icorp = [(0, image_id) for image_id in s_testfiles_den]
    filelist_w_icorp.extend([(1, image_id) for image_id in r_testfiles_den])

    X_full = np.concatenate([Xsaia, Xcoco])
    #X_full = Xsaia

    with gzip.open('../../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
        s_bbdf = pickle.load(f)
    s_bbdf = filter_by_filelist(s_bbdf, s_testfiles)

    with gzip.open('../../Preproc/PreProcOut/mscoco_bbdf.pklz', 'r') as f:
        c_bbdf = pickle.load(f)
    c_bbdf = filter_by_filelist(c_bbdf, r_testfiles)

    bbdf = pd.concat([s_bbdf, c_bbdf])
    #bbdf = s_bbdf


    with gzip.open('../../Preproc/PreProcOut/refcoco_refdf.pklz', 'r') as f:
        cocrefdf = pickle.load(f)
    with gzip.open('../../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
        srefdf = pickle.load(f)
    refdf = pd.concat([srefdf, cocrefdf])
    #refdf = srefdf
    #***

    #if stopped:
    #    wlist = [word for word in wac_05.keys() if not word in ['on','to','from','of','at','in','with','us','is','you','what']]
     #   word2ind = dict([(word, no) for no, word in enumerate(wlist)])

    #else:
    word2ind = dict([(word, no) for no, word in enumerate(wac_05.keys())])
    wlist = [word for word in wac_05.keys()]

    
    print "Set up set_up_lindenotational_similarity"

    linden_mat,linden_vecs = set_up_lindenotational_similarity(X_full, bbdf, filelist_w_icorp,
                                   linwac, wlist, nobj=NTEST)

    linint_mat = set_up_linintensional_similarity(linwac, wlist)

   
    return linden_mat, linint_mat,linden_vecs


def load_linmat_unk(stopped=False):

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/saiapr.npz')
    Xsaia = X['arr_0']

    print "Xsaia", Xsaia.shape

    

    X = np.load(DATADIRPREF + '/Models/2016-v3-image-wac/mscoco.npz')
    Xcoco = X['arr_0']

    with gzip.open('../linmodels/linwac_unks_w2v_semeval_nneg0.1.pklz', 'r') as f:
        linwac = pickle.load(f)

    with open('../../Preproc/PreProcOut/refcoco_splits.json', 'r') as f:
        rcocosplits = json.load(f)
    r_testfiles = rcocosplits['testB']

    with open('../../Preproc/PreProcOut/saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)
    s_testfiles = ssplit90['test']

    s_testfiles = s_testfiles[:-NTEST]
    s_testfiles_den = s_testfiles[-NTEST:]
    r_testfiles_den = r_testfiles[:NTEST]  # could use testA, but have filtered bbdf above..
    filelist_w_icorp = [(0, image_id) for image_id in s_testfiles_den]
    filelist_w_icorp.extend([(1, image_id) for image_id in r_testfiles_den])

    X_full = np.concatenate([Xsaia, Xcoco])
    #X_full = Xsaia

    with gzip.open('../../Preproc/PreProcOut/saiapr_bbdf.pklz', 'r') as f:
        s_bbdf = pickle.load(f)
    s_bbdf = filter_by_filelist(s_bbdf, s_testfiles)

    with gzip.open('../Preproc/PreProcOut/mscoco_bbdf.pklz', 'r') as f:
        c_bbdf = pickle.load(f)
    c_bbdf = filter_by_filelist(c_bbdf, r_testfiles)

    bbdf = pd.concat([s_bbdf, c_bbdf])
    #bbdf = s_bbdf


    
    #else:
    word2ind = dict([(word, no) for no, word in enumerate(linwac.keys())])
    wlist = [word for word in linwac.keys()]

    
    print "Set up set_up_lindenotational_similarity"

    linden_mat,linden_vecs = set_up_lindenotational_similarity(X_full, bbdf, filelist_w_icorp,
                                   linwac, wlist, nobj=NTEST)

   
    return linden_mat, linden_vecs, word2ind, wlist


def set_up_visual_similarity(wlist,Xsaia,Xcoco,Wsaia,Wcoco):


    veclist = []
    for x,w in enumerate(wlist):
        
        wsample_coco = Wcoco[x]
            #print list(wsample_coco).count(1)
        wsample_saia = Wsaia[x]
            #print list(wsample_saia).count(1)
        fcoco = Xcoco[wsample_coco == 1][:,3:]
            #print "fcoco", fcoco.shape
        fsaia = Xsaia[wsample_saia == 1][:,3:]
            #print "fsaia", fsaia.shape

        avvec = np.mean(np.vstack([fcoco,fsaia]),axis=0)
            #print "avvec", avvec.shape
        veclist.append(avvec)
    

    avmat = np.array(veclist)
    sim_matr = scipy.spatial.distance.pdist(avmat, 'cosine')
    sim_matr_sq = scipy.spatial.distance.squareform(sim_matr)

    return avmat,sim_matr_sq


def set_up_label_similarity(wlist,Xsaia,Wsaia):


    veclist = []
    for x,w in enumerate(wlist):
        
        
        wsample_saia = Wsaia[x]
            #print list(wsample_saia).count(1)
        fsaia = Xsaia[wsample_saia == 1][:,3:]
            #print "fsaia", fsaia.shape

        avvec = np.mean(fsaia,axis=0)
            #print "avvec", avvec.shape
        veclist.append(avvec)
    

    avmat = np.array(veclist)
    sim_matr = scipy.spatial.distance.pdist(avmat, 'cosine')
    sim_matr_sq = scipy.spatial.distance.squareform(sim_matr)
    
    

    return sim_matr_sq

def set_up_w2v_similarity(wlist):

    i = 0
    badict = {}
    for line in open('../Data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt'):
        i += 1
        l = line.split()
        w = l[0]
        if w in wlist:
            badict[w] = l[1:]

    tlen = len(badict[badict.keys()[0]])
    balist = []
    for w in wlist:
        if w in badict:
            balist.append(badict[w])
        else:
            balist.append([0]*tlen) 

    mat = np.array(balist)
    sim_matr = scipy.spatial.distance.pdist(mat, 'cosine')
    sim_matr_sq = scipy.spatial.distance.squareform(sim_matr)
    
    return mat,sim_matr_sq


def set_up_lindenotational_similarity(X, bbdf, filelist_w_icorp,
                                   linwac, wordlist, nobj=2000):
    cols = ['i_corpus', 'image_id']
    den_bbdf = bbdf.merge(pd.DataFrame(filelist_w_icorp, columns=cols), on=cols)
    subsample = den_bbdf.iloc[np.random.choice(range(len(den_bbdf)), nobj)][['i_corpus', 'image_id', 'region_id']]
    denot_X = np.concatenate([X[np.logical_and(X[:,0] == row['i_corpus'],
                                               np.logical_and(X[:,1] == row['image_id'],
                                                              X[:,2] == row['region_id']))] 
                              for _, row in subsample.iterrows()])
    denot_X = denot_X[:,3:]
    denot_vec = []
    for word in wordlist:
        if linwac[word]:
            #denot_vec.append([1-sigmoid(x) for x in linwac[word].predict(denot_X)])
            denot_vec.append(linwac[word].predict(denot_X))
        else:
            denot_vec.append(np.array([np.nan]*len(denot_X)))
    word_vecs_den = np.array(denot_vec)
    print word_vecs_den.shape
    sim_matr_den = scipy.spatial.distance.pdist(word_vecs_den, 'cosine')
    sim_matr_sq_den = scipy.spatial.distance.squareform(sim_matr_den)
    return sim_matr_sq_den,word_vecs_den

def set_up_linintensional_similarity(wac,wordlist):
    
    veclen = len(wac[wac.keys()[0]].coef_)

    word_vecs_int = []
    for word in wordlist:
        if wac[word]:
            word_vecs_int.append(wac[word].coef_)
        else:
            word_vecs_int.append([np.nan]*veclen)

    sim_matr_int = scipy.spatial.distance.pdist(word_vecs_int, 'cosine')
    sim_matr_sq_int = scipy.spatial.distance.squareform(sim_matr_int)
    
    return sim_matr_sq_int

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def turn_list_into_distr(rawlist):
    sum_ = len(rawlist)
    return [np.log(e/sum_) for e in rawlist]



def get_coocurrences(refdf, wordlist, filelist):
    noun_singles = []
    noun_pairs_overall = []
    for (this_corpus,this_file) in tqdm(filelist):
        this_subdf = refdf.query('(i_corpus == %d) & (image_id == %d)' % (this_corpus,this_file))
        nounlist = [w for w in ' '.join(this_subdf['refexp'].tolist()).split() if w in wordlist]

        noun_singles.extend(nounlist)
        
        all_region_pairs = list(combinations(set(this_subdf['region_id'].tolist()), 2))
        
        noun_pairs = []
        for left_region, right_region in all_region_pairs:
            left_df = this_subdf.query('region_id == %d' % (left_region))
            left_nouns = [w for w in ' '.join(left_df['refexp'].tolist()).split() if w in wordlist]

            right_df = this_subdf.query('region_id == %d' % (right_region))
            right_nouns = [w for w in ' '.join(right_df['refexp'].tolist()).split() if w in wordlist]

            noun_pairs.extend([(left_noun, right_noun)\
                               for left_noun in left_nouns\
                               for right_noun in right_nouns])
        noun_pairs = [(e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in noun_pairs]
        
        noun_pairs_overall.extend(noun_pairs)
    return noun_singles, noun_pairs_overall


def turn_into_distr(counter, rawlist):
    sum_ = len(rawlist)
    return dict([(e[0], np.log(e[1]/sum_)) for e in counter.items()])


def pairwise_mi(w1, w2, single_dist, pair_dist):
    if (w1,w2) not in pair_dist:
        w1, w2 = w2, w1
    if (w1,w2) not in pair_dist:
        return 0
    return np.exp(pair_dist[(w1,w2)] - (single_dist[w1] + single_dist[w2])) / - pair_dist[(w1,w2)]



def set_up_context_associations(refdf, wordlist, filelist):
    noun_singles, noun_pairs = get_coocurrences(refdf, wordlist, filelist)

    single_counter = Counter(noun_singles)
    pair_counter = Counter(noun_pairs)


    single_dist = turn_into_distr(single_counter, noun_singles)
    pair_dist = turn_into_distr(pair_counter, noun_pairs)

    assert sorted(wordlist) == sorted(single_dist.keys())
    ind_X = np.arange(len(wordlist)).reshape(len(wordlist), 1)


    pmi_mat = pdist(ind_X, lambda x,y: pairwise_mi_by_ind(x,y, wordlist, single_dist, pair_dist))
    print "PMI mat", pmi_mat.shape
    pmi_sq = squareform(pmi_mat)
    print "PMI sq", pmi_sq.shape

    sim_matr_pmi = scipy.spatial.distance.pdist(pmi_sq, 'cosine')
    sim_matr_pmi_sq = squareform(sim_matr_pmi)

    return pmi_sq, sim_matr_pmi_sq

class MSimilarity(object):

    def __init__(self, den, vis, w2v, context, word2ind, linden, wlist):

        self.word2ind = word2ind
        self.wordlist = wlist
        (self.visual,self.visual_vecs) = vis[0],vis[1]
        (self.extensional,self.ext_vecs) = den[0],den[1]
        (self.w2v,self.w2v_vecs) = w2v[0],w2v[1] 
        (self.context_sq,self.context_pmi) = context[0],context[1]
        self.ext_onto = linden

class UnkSimilarity(object):

    def __init__(self, den_vecs, word2ind, wlist):

        self.word2ind = word2ind
        self.wordlist = wlist
        self.ext_vecs = den_vecs

if __name__ == '__main__':

    

    with gzip.open('../../Data/multi_similarity2.pklz', 'r') as f:
        msim = pickle.load(f)

    ontoden,ontoint,ontovecs = load_linmat()
    msim.ext_onto = ontoden
    msim.ext_onto_vecs = ontovecs

    with gzip.open('../../Data/multi_similarity3.pklz', 'w') as f:
        pickle.dump(msim,f)

    #simmat,vecs,word2ind,wordlist = load_linmat_unk()
    #unksim = UnkSimilarity(vecs,word2ind,wordlist)
    #with gzip.open('../Data/unk_similarity.pklz', 'w') as f:
    #    pickle.dump(unksim,f)

