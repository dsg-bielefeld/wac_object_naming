from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json,yaml
import os
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import sys
import random

from utils import filter_by_filelist, filter_X_by_filelist
from utils import STOPWORDS, is_relational
sys.path.append('..')
from apply_model import *
import linwac
import linmap
import logwac





with gzip.open('../indata/multi_similarity2.pklz', 'r') as f:
    msim = pickle.load(f)

noun_list = [l.strip() for l in open('../indata/noun_list.txt').readlines()]
noun_ind = [msim.word2ind[n] for n in noun_list]

random.shuffle(noun_list)

def make_saia_noun_splits():

    print len(noun_list)
    print noun_list[:10]

    splits = range(0,len(noun_list),int(len(noun_list)/10))
    noun_splits = []

    for i,s in enumerate(splits):
        
        if i != 10:
            print i,s, noun_list[s:splits[i+1]]
            noun_splits.append(noun_list[s:splits[i+1]])
        else:
            print i,s, noun_list[s:]
            noun_splits.append(noun_list[s:])

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    ttsplit = []
    for nspl in noun_splits:
        testspl = []
        for w in nspl:
            wx = msim.word2ind[w]
            testspl += list(Wsaia[wx].nonzero()[0])

        trainspl = [i for i in range(Wsaia.shape[1]) if not i in testspl]
        print "Train/test", len(trainspl), len(testspl)

        ttsplit.append({'train':trainspl,'test':testspl,'nouns':nspl})


    with open('../indata/saia_zeroshot_nounsplits.json', 'w') as f:
        json.dump(ttsplit,f)

def make_saia_hyper_split():

    hyper = ['animal','animals','plant','plants','vehicle','person','persons','food','thing','object','area',\
    'things','thingy','toy','anyone','clothes','dish','building','land','structure','item','water']

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    testspl = []
    for hyperw in hyper:
        wx = msim.word2ind[hyperw]
        testspl += list(Wsaia[wx].nonzero()[0])

    trainspl = [i for i in range(Wsaia.shape[1]) if not i in testspl]
    print "Train/test", len(trainspl), len(testspl)


    with open('../indata/saia_zeroshot_hypernsplit.json', 'w') as f:
        json.dump({'train':trainspl,'test':testspl,'nouns':hyper},f)


def make_plural_splits():

    plurals = ['animals','plants','cars','people','buildings','trees','women','men','kids','guys','girls','boys','flowers','birds','hills','oranges','clouds',\
    'curtains','windows','shrubs','apples','lights','houses','glasses','bottles','dudes','legs','books','walls','bananas','carrots','pillows','bushes','mountains','bags']
    singulars = ['animal','plant','car','person','building','tree','woman','man','kid','guy','girl','boy','flower','bird','hill','orange','cloud',\
    'curtain','window','shrub','apple','light','house','glass','bottle','dude','leg','book','wall','banana','carrot','pillow','bush','mountain','bag']

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    testspl = []
    trainspl = []
    saia_singulars = []
    saia_plurals = []
    for x,singword in enumerate(singulars):
        sing_x = msim.word2ind[singword]
        plu_x = msim.word2ind[plurals[x]]
        pos_ind_sing = list(Wsaia[sing_x].nonzero()[0])
        pos_ind_plu = list(Wsaia[plu_x].nonzero()[0])

        if (len(pos_ind_sing) > 0) and (len(pos_ind_plu) > 0):
            testspl += pos_ind_plu
            saia_singulars.append(singword)
            saia_plurals.append(plurals[x])

    testspl = list(set(testspl))

    for singword in saia_singulars:
        sing_x = msim.word2ind[singword]
        pos_ind_sing = list(Wsaia[sing_x].nonzero()[0])
        trainspl += [pos_i for pos_i in pos_ind_sing if not pos_i in testspl]

    trainspl = list(set(trainspl))


    
    print "Saia Train/test", len(trainspl), len(testspl)
    print "saia_singulars", saia_singulars


    with open('../indata/saia_zeroshot_pluralsplit.json', 'w') as f:
        json.dump({'train':trainspl,'test':testspl,'nouns':saia_plurals,'singulars':saia_singulars},f)


def make_standard_plural_split():

    plurals = ['animals','plants','cars','people','buildings','trees','women','men','kids','guys','girls','boys','flowers','birds','hills','oranges','clouds',\
    'curtains','windows','shrubs','apples','lights','houses','glasses','bottles','dudes','legs','books','walls','bananas','carrots','pillows','bushes','mountains','bags']
    singulars = ['animal','plant','car','person','building','tree','woman','man','kid','guy','girl','boy','flower','bird','hill','orange','cloud',\
    'curtain','window','shrub','apple','light','house','glass','bottle','dude','leg','book','wall','banana','carrot','pillow','bush','mountain','bag']

    with open('../indata/saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)


    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    Xsaia_train = Xsaia[np.in1d(Xsaia[:,1],ssplit90['train'])]


    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape


    Wsaia_t_train = Wsaia_t[np.in1d(Xsaia[:,1],ssplit90['train'])]
    Wsaia_train = Wsaia_t_train.transpose()

    Wsaia_t_test = Wsaia_t[np.in1d(Xsaia[:,1],ssplit90['test'])]
    Wsaia_test = Wsaia_t_test.transpose()

    print "Wsaia", Wsaia.shape

    testspl = []
    trainspl = []
    saia_nouns = []
    for x,singword in enumerate(singulars):
        sing_x = msim.word2ind[singword]
        plu_x = msim.word2ind[plurals[x]]
        pos_ind_sing = list(Wsaia_train[sing_x].nonzero()[0])
        pos_ind_plu = list(Wsaia_train[plu_x].nonzero()[0])
        pos_ind_plu_test = list(Wsaia_test[plu_x].nonzero()[0])

        if (len(pos_ind_plu_test) > 0) and (len(pos_ind_plu) > 0):
            trainspl += pos_ind_sing + pos_ind_plu
            testspl += pos_ind_plu_test

            saia_nouns.append(singword)
            saia_nouns.append(plurals[x])

    
    trainspl = list(set(trainspl))
    testspl = sorted(list(set(testspl)))

    old_testspl = np.where(np.in1d(Xsaia[:,1],ssplit90['test']))[0]
    new_testspl = list(old_testspl[testspl])

    print testspl[:10]
    print old_testspl[:10]
    print new_testspl[:10]

    print "Saia Train/test", len(trainspl), len(testspl)
    print "saia_nouns", len(saia_nouns)


    with open('../indata/saia_standard_pluralsplit.json', 'w') as f:
        json.dump({'train':trainspl,'test':new_testspl,'nouns':saia_nouns},f)


def make_mixed_plural_splits():

    plurals = ['animal','plant','car','person','building','tree','woman','women','men','kids','guys','girls','boys','flowers','birds','hills','oranges','clouds',\
    'curtains','windows','shrubs','apples','lights','houses','glasses','bottles','dudes','legs','books','walls', 'banana','carrot','pillow','bush','mountain','bag']
    singulars = ['animals','plants','cars','people','buildings','trees','man','kid','guy','girl','boy','flower','bird','hill','orange','cloud',\
    'curtain','window','shrub','apple','light','house','glass','bottle','dude','leg','book','wall','bananas','carrots','pillows','bushes','mountains','bags']

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    testspl = []
    trainspl = []
    saia_singulars = []
    saia_plurals = []
    for x,singword in enumerate(singulars):
        sing_x = msim.word2ind[singword]
        plu_x = msim.word2ind[plurals[x]]
        pos_ind_sing = list(Wsaia[sing_x].nonzero()[0])
        pos_ind_plu = list(Wsaia[plu_x].nonzero()[0])

        if (len(pos_ind_sing) > 0) and (len(pos_ind_plu) > 0):
            testspl += pos_ind_plu
            saia_singulars.append(singword)
            saia_plurals.append(plurals[x])

    testspl = list(set(testspl))

    for singword in saia_singulars:
        sing_x = msim.word2ind[singword]
        pos_ind_sing = list(Wsaia[sing_x].nonzero()[0])
        trainspl += [pos_i for pos_i in pos_ind_sing if not pos_i in testspl]

    trainspl = list(set(trainspl))


    
    print "Saia Train/test", len(trainspl), len(testspl)
    print "saia_singulars", saia_singulars


    with open('../indata/saia_zeroshot_mixedpluralsplit.json', 'w') as f:
        json.dump({'train':trainspl,'test':testspl,'nouns':saia_plurals,'singulars':saia_singulars},f)



def make_coco_hyper_split():

    hyper = ['animal','animals','plant','plants','vehicle','person','persons','food','thing','object','area',\
    'things','thingy','toy','anyone','clothes','dish','building','land','structure','item','water']

    W = np.load('../indata/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']

    print "Wcoco", Wcoco.shape

    testspl = []
    for hyperw in hyper:
        wx = msim.word2ind[hyperw]
        testspl += list(Wcoco[wx].nonzero()[0])

    trainspl = [i for i in range(Wcoco.shape[1]) if not i in testspl]
    print "Train/test", len(trainspl), len(testspl)


    with open('../indata/coco_zeroshot_hypernsplit.json', 'w') as f:
        json.dump({'train':trainspl,'test':testspl,'nouns':hyper},f)


def make_saia_noun_long_splits():

    long_noun_list = [l.strip() for l in open('noun_list_long.txt').readlines()]
    print len(long_noun_list)
    print long_noun_list[:10]
    random.shuffle(long_noun_list)

    splits = range(0,len(long_noun_list),int(len(long_noun_list)/5))
    print splits
    noun_splits = []

    for i,s in enumerate(splits):
        
        if i != 4:
            print i,s, long_noun_list[s:splits[i+1]]
            noun_splits.append(long_noun_list[s:splits[i+1]])
        else:
            print i,s, long_noun_list[s:]
            noun_splits.append(long_noun_list[s:])
            break

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']

    print "Wsaia", Wsaia.shape

    ttsplit = []
    for nspl in noun_splits:
        testspl = []
        for w in nspl:
            wx = msim.word2ind[w]
            testspl += list(Wsaia[wx].nonzero()[0])

        trainspl = [i for i in range(Wsaia.shape[1]) if not i in testspl]
        print "Train/test", len(trainspl), len(testspl)

        ttsplit.append({'train':trainspl,'test':testspl,'nouns':nspl})


    with open('../indata/saia_zeroshot_nounslong_splits.json', 'w') as f:
        json.dump(ttsplit,f)

def make_coco_noun_splits():

    long_noun_list = [l.strip() for l in open('noun_list_long.txt').readlines()]
    log_noun_ind = [msim.word2ind[n] for n in long_noun_list]

    random.shuffle(long_noun_list)

    splits = range(0,len(long_noun_list),int(len(long_noun_list)/5))
    print splits
    noun_splits = []

    for i,s in enumerate(splits):
        if i != 4:
            print i,s, long_noun_list[s:splits[i+1]]
            noun_splits.append(long_noun_list[s:splits[i+1]])
        else:
            print i,s, long_noun_list[s:]
            noun_splits.append(long_noun_list[s:])
            break

    W = np.load('../indata/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']

    print "Wcoco", Wcoco.shape

    cocoref = [rx for rx in range(Wcoco.shape[1]) if len(np.nonzero(Wcoco[:,rx])[0]) > 0]
    print "Total ref", len(cocoref)

    ttsplit = []
    for nspl in noun_splits:
        testspl = []
        for w in nspl:
            wx = msim.word2ind[w]
            testspl += list(Wcoco[wx].nonzero()[0])

        trainspl = [i for i in cocoref if not i in testspl]
        print "Train/test", len(trainspl), len(testspl)

        ttsplit.append({'train':trainspl,'test':testspl,'nouns':nspl})


    with open('../indata/refcoco_zeroshot_nounsplits.json', 'w') as f:
        json.dump(ttsplit,f)


def count_coco_nouns():

    long_noun_list = [l.strip() for l in open('noun_list_long.txt').readlines()]
    total = 0

    W = np.load('../indata/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']

    for w in long_noun_list:
        wx = msim.word2ind[w]
        pos_ind = list(Wcoco[wx].nonzero()[0])
        if len(pos_ind) > 0:
            total +=1
        else:
            print w

    print "Nouns with positive instances", total


def train_zero_saia_models(word_list=noun_list):


    with open('../indata/saia_zeroshot_nounsplits.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape

    

    for x,spl in enumerate(ttsplit):
        print "SPLIT",x

        if x > 0:

            Xsaia_train = Xsaia[spl['train']]
            Xsaia_test = Xsaia[spl['test']]

            Wsaia_t = Wsaia.transpose()
            Wsaia_t.shape


            Wsaia_t_train = Wsaia_t[spl['train']]
            Wsaia_train = Wsaia_t_train.transpose()

            print "Train linwac"
            linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="_zeroshot_split"+str(x))

            print "Train transfer"
            linmap.train_mappings(msim.w2v_vecs,noun_ind,Wsaia_train,Xsaia_train,split="_zeroshot_split"+str(x))

            print "Train logwac"
            logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=noun_list,ssim="nouns_zeroshot_split"+str(x))


def train_zero_hypern_saia_models():


    with open('../indata/saia_zeroshot_hypernsplit.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape

    

    Xsaia_train = Xsaia[ttsplit['train']]
    Xsaia_test = Xsaia[ttsplit['test']]

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape
    Wsaia_t_train = Wsaia_t[ttsplit['train']]
    Wsaia_train = Wsaia_t_train.transpose()

    print ttsplit['nouns']

    print "Train linwac"
    this_wordlist = noun_list + [n for n in ttsplit['nouns'] if not n in noun_list]
    print "Wordlist", len(this_wordlist)
    linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="_zeroshot_hypernsplit",word_list=this_wordlist)

    print "Train transfer"
    linmap.train_mappings(msim.w2v_vecs,noun_ind,Wsaia_train,Xsaia_train,split="_zeroshot_hypernsplit")

    print "Train logwac"
    logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=noun_list,ssim="nouns_zeroshot_hypernsplit")

def train_zero_plural_saia_models():


    with open('../indata/saia_zeroshot_pluralsplit.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape


    Xsaia_train = Xsaia[ttsplit['train']]
    Xsaia_test = Xsaia[ttsplit['test']]

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape
    Wsaia_t_train = Wsaia_t[ttsplit['train']]
    Wsaia_train = Wsaia_t_train.transpose()

    print "Plurals",ttsplit['nouns']
    print "Singulars",ttsplit['singulars']

    print "Train linwac"
    this_wordlist = ttsplit['nouns'] + ttsplit['singulars']
    word_ind = [msim.word2ind[n] for n in this_wordlist]

    print "Wordlist", len(this_wordlist)
    linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="_zeroshot_pluralsplit",word_list=this_wordlist)

    print "Train transfer"
    linmap.train_mappings(msim.w2v_vecs,word_ind,Wsaia_train,Xsaia_train,split="_zeroshot_pluralsplit")

    print "Train logwac"
    logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=this_wordlist,ssim="nouns_zeroshot_pluralsplit")

def train_zero_mixed_plural_saia_models():


    with open('../indata/saia_zeroshot_mixedpluralsplit.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape


    Xsaia_train = Xsaia[ttsplit['train']]
    Xsaia_test = Xsaia[ttsplit['test']]

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape
    Wsaia_t_train = Wsaia_t[ttsplit['train']]
    Wsaia_train = Wsaia_t_train.transpose()

    print "Plurals",ttsplit['nouns']
    print "Singulars",ttsplit['singulars']

    print "Train linwac"
    this_wordlist = ttsplit['nouns'] + ttsplit['singulars']
    word_ind = [msim.word2ind[n] for n in this_wordlist]

    print "Wordlist", len(this_wordlist)
    linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="_zeroshot_mixedpluralsplit",word_list=this_wordlist)

    print "Train transfer"
    linmap.train_mappings(msim.w2v_vecs,word_ind,Wsaia_train,Xsaia_train,split="_zeroshot_mixedpluralsplit")

    print "Train logwac"
    logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=this_wordlist,ssim="nouns_zeroshot_mixedpluralsplit")

def train_standard_plural_saia_models():


    with open('../indata/saia_standard_pluralsplit.json', 'r') as f:
        ttsplit = json.load(f)

    print "Nouns", len(ttsplit['nouns'])

    w2v = linwac.load_w2v()


    Xsaia_t,Wsaia_t = linwac.load_saia_train()

    Xsaia_train = Xsaia_t[ttsplit['train']]

    Wsaia_tt = Wsaia_t.transpose()
    print Wsaia_tt.shape
    Wsaia_t_train = Wsaia_tt[ttsplit['train']]
    Wsaia_train = Wsaia_t_train.transpose()


    print "Train linwac"
    this_wordlist = ttsplit['nouns']
    word_ind = [msim.word2ind[n] for n in this_wordlist]

    print "Wordlist", len(this_wordlist)
    linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="_standard_pluralsplit",word_list=this_wordlist)

    print "Train transfer"
    linmap.train_mappings(msim.w2v_vecs,word_ind,Wsaia_train,Xsaia_train,split="_standard_pluralsplit")

    print "Train logwac"
    logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=this_wordlist,ssim="nouns_standard_pluralsplit")


def train_zero_saia_longlist(word_list=noun_list):


    with open('../indata/saia_zeroshot_nounslong_splits.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape

    long_noun_list = [l.strip() for l in open('noun_list_long.txt').readlines()]
    long_noun_ind = [msim.word2ind[n] for n in long_noun_list]

    for x,spl in enumerate(ttsplit):
        print "SPLIT",x

        if x > 0:

            Xsaia_train = Xsaia[spl['train']]
            Xsaia_test = Xsaia[spl['test']]

            Wsaia_t = Wsaia.transpose()
            Wsaia_t.shape


            Wsaia_t_train = Wsaia_t[spl['train']]
            Wsaia_train = Wsaia_t_train.transpose()

            print "Train linwac"
            linwac.train_all_nouns(Wsaia_train,Xsaia_train,w2v,ssim="500n_zeroshot_split"+str(x),word_list=long_noun_list)

            print "Train transfer"
            linmap.train_mappings(msim.w2v_vecs,long_noun_ind,Wsaia_train,Xsaia_train,split="500n_zeroshot_split"+str(x))

            print "Train logwac"
            logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=noun_list,ssim="nouns_zeroshot_split"+str(x))


def train_zero_refcoco_models():

    long_noun_list = [l.strip() for l in open('noun_list_long.txt').readlines()]
    long_noun_ind = [msim.word2ind[n] for n in long_noun_list]


    with open('../indata/refcoco_zeroshot_nounsplits.json', 'r') as f:
        ttsplit = json.load(f)

    w2v = linwac.load_w2v()

    X = np.load('../indata/mscoco.npz')
    Xcoco = X['arr_0']
    print "Xcoco", Xcoco.shape

    W = np.load('../indata/mscoco_refcoco_wmat.npz')
    Wcoco = W['arr_0']
    print "Wcoco", Wcoco.shape

    total = 0
    for rx in range(Wcoco.shape[1]):
        if len(np.nonzero(Wcoco[:,rx])[0]):
            total += 1
    print "Total", total

    with gzip.open('../indata/refcoco_refdf.pklz', 'r') as f:
        refdf = pickle.load(f)

    print "Regionids", len(set(refdf['region_id']))

    for x,spl in enumerate(ttsplit):
        print "SPLIT",x

        if x == 2:

            Xcoco_train = Xcoco[spl['train']]
            Xcoco_test = Xcoco[spl['test']]

            print "Xcoco",Xcoco_train.shape

            Wcoco_t = Wcoco.transpose()
            Wcoco_t.shape
            Wcoco_t_train = Wcoco_t[spl['train']]
            Wcoco_train = Wcoco_t_train.transpose()

            print "Wcoco", Wcoco_train.shape

            total = 0
            for rx in range(Wcoco_train.shape[1]):
                if len(np.nonzero(Wcoco_train[:,rx])[0]):
                    total += 1
            print "Total", total
               

            print "Train linwac"
            linwac.train_vocab_abssample(Wcoco_train,Xcoco_train,w2v,\
                nsamp=Xcoco_train.shape[0],scorp="rcoco",ssim="w2v_zeroshot_split"+str(x),word_list=long_noun_list)
            #linwac.train_all_nouns(Wcoco_train,Xcoco_train,w2v,ssim="refcoco_w2v_zeroshot_split"+str(x),word_list=long_noun_list)

            print "Train transfer"
            linmap.train_mappings(msim.w2v_vecs,long_noun_ind,Wcoco_train,Xcoco_train,split="_refcoco_zeroshot_split"+str(x))

            print "Train logwac"
            logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=noun_list,ssim="nouns_zeroshot_split"+str(x))


def train_zero_saia_min10words(word_list):


    with open('../indata/saia_zeroshot_nounsplits.json', 'r') as f:
        ttsplit = json.load(f)

    X = np.load('../indata/saiapr.npz')
    Xsaia = X['arr_0']
    print "Xsaia", Xsaia.shape

    W = np.load('../indata/saiapr_wmat.npz')
    Wsaia = W['arr_0']
    print "Wsaia", Wsaia.shape

    

    split = ttsplit[1]
    print "Unknown",split['nouns']


    Xsaia_train = Xsaia[split['train']]
    Xsaia_test = Xsaia[split['test']]

    Wsaia_t = Wsaia.transpose()
    Wsaia_t.shape


    Wsaia_t_train = Wsaia_t[split['train']]
    Wsaia_train = Wsaia_t_train.transpose()

    train_wordlist = []
    for w in word_list:
        wx = msim.word2ind[w]
        print w,wx
        if len(np.where(Wsaia_train[wx] == 1)[0]) > 10:
            train_wordlist.append(w)

    print "Words occuring more than 10 times", len(train_wordlist), "out of", len(word_list)


    print "Train logwac"
    logwac.train_saia_nosamp(Xsaia_train,Wsaia_train,word_list=train_wordlist,ssim="min10words_zeroshot_split"+str(1))




if __name__ == '__main__':

    #make_coco_noun_splits()

    #make_saia_noun_long_splits()

    #make_saia_hyper_split()

    #make_plural_splits()

    #make_mixed_plural_splits()

    #make_standard_plural_split()

    #make_coco_hyper_split()

    train_zero_saia_models()

    train_zero_saia_longlist()

    train_zero_hypern_saia_models()

    train_zero_plural_saia_models()

    train_zero_mixed_plural_saia_models()

    train_standard_plural_saia_models()

   


