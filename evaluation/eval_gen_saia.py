from __future__ import division
import pandas as pd
import numpy as np
import gzip
import cPickle as pickle
import json
import itertools
from tqdm import tqdm
from collections import Counter
import yaml
import sys


#from utils import filter_by_filelist, filter_X_by_filelist, filter_by_fileregionlist
import utils
from utils import STOPWORDS, is_relational
sys.path.append('../training')
from msimilarity import MSimilarity
import linwac
from apply_model_matrix import *


NOUNS = [l.strip() for l in open('../indata/noun_list.txt').readlines()]

with gzip.open('../indata/multi_similarity2.pklz', 'r') as f:
    msim = pickle.load(f)

X = np.load('../indata/saiapr.npz')
Xsaia = X['arr_0']
print "Xsaia", Xsaia.shape

def apply_map_model(testset,model_path,word_mat,word_list):

    word_dict = {n: word_mat[x] for x,n in enumerate(word_list)}

   
    (tst_index,tst_out) = testset[0]
    Xsaia_test = Xsaia[tst_index,3:]
    print "Nouns",word_list

    with gzip.open(model_path, 'r') as f:
        mapmodel = pickle.load(f)

    projected_words,words_prob = decode_transfer_prob(Xsaia_test,mapmodel,word_mat,word_list,10)
    get_accuracy(tst_out,projected_words)
    return projected_words,words_prob


def apply_lin_model(testset,model_path,word_list):

   
    (tst_index,tst_out) = testset[0]
    Xsaia_test = Xsaia[tst_index,3:]

    with gzip.open(model_path, 'r') as f:
        wacmodel = pickle.load(f)

    pred_words,words_probs = get_topn_linwacs(Xsaia_test,wacmodel,10,nounlist=word_list)
    get_accuracy(tst_out,pred_words)

    return pred_words,words_probs


def apply_log_model(testset,model_path,word_list):

   
    (tst_index,tst_out) = testset[0]
    Xsaia_test = Xsaia[tst_index,3:]

    with gzip.open(model_path, 'r') as f:
        wacmodel = pickle.load(f)

    pred_words,words_probs = get_topn_logwacs(Xsaia_test,wacmodel,10,nounlist=word_list)

    get_accuracy(tst_out,pred_words)
    return pred_words,words_probs

def apply_project_log_model(testset,model_path,word_mat,word_list):

   
    (tst_index,tst_out) = testset[0]
    Xsaia_test = Xsaia[tst_index,3:]

    with gzip.open(model_path, 'r') as f:
        wacmodel = pickle.load(f)

    pred_words,words_probs = get_topn_logwacs(Xsaia_test,wacmodel,10,nounlist=word_list)

    projected_words10 = decode_and_map_wac(Xsaia_test,wacmodel,word_mat,word_list,word_list,10,10)
    projected_words5 = decode_and_map_wac(Xsaia_test,wacmodel,word_mat,word_list,word_list,5,10)

    print "Project 10 words"
    get_accuracy(tst_out,projected_words10)
    print "Project 5 words"
    get_accuracy(tst_out,projected_words5)


    return True

def combine_topn_maplinlog((word_mat1,prob_mat1),(word_mat2,prob_mat2),(word_mat3,prob_mat3)):

    
    prob_mat1_inv = 1-prob_mat1
    totals_1 = np.sum(prob_mat1_inv,axis=1)
    normprob_mat1 = prob_mat1_inv/totals_1[:,None]
  
    #print prob_mat2[:2]
    prob_mat2_inv = prob_mat2*-1
    #print prob_mat2_inv[:2]
    totals_2 = np.sum(prob_mat2_inv,axis=1)
    #print totals_2[:2]
    normprob_mat2 = prob_mat2_inv/totals_2[:,None]
    #print normprob_mat2[:2]

    #print "***"
    #print prob_mat3[:2]
    totals_3 = np.sum(prob_mat3,axis=1)
    normprob_mat3 = prob_mat3/totals_3[:,None]
    #print normprob_mat3[:2]

    #for i in range(len(word_mat1)):
    gen_comb = []
    for i in range(len(word_mat1)):
    #for i in range(10):
        comb = Counter()
        for x,w in enumerate(word_mat1[i]):
            comb[w] += normprob_mat1[i][x]
        for x,w in enumerate(word_mat2[i]):
            comb[w] += normprob_mat2[i][x]
        for x,w in enumerate(word_mat3[i]):
            comb[w] += normprob_mat3[i][x]

        #print "Map", word_mat1[i][:5]
        #print "Lin", word_mat2[i][:5]
        #print "Log", word_mat3[i][:5]
        #print comb.most_common(10)
        gen_comb.append([w for (w,_) in comb.most_common(10)])

    return gen_comb


def combine_topn_maplin((word_mat1,prob_mat1),(word_mat2,prob_mat2)):

    
    prob_mat1_inv = 1-prob_mat1
    totals_1 = np.sum(prob_mat1_inv,axis=1)
    normprob_mat1 = prob_mat1_inv/totals_1[:,None]
  
    #print prob_mat2[:2]
    prob_mat2_inv = prob_mat2*-1
    #print prob_mat2_inv[:2]
    totals_2 = np.sum(prob_mat2_inv,axis=1)
    #print totals_2[:2]
    normprob_mat2 = prob_mat2_inv/totals_2[:,None]
    #print normprob_mat2[:2]


    #for i in range(len(word_mat1)):
    gen_comb = []
    for i in range(len(word_mat1)):
    #for i in range(10):
        comb = Counter()
        for x,w in enumerate(word_mat1[i]):
            comb[w] += normprob_mat1[i][x]
        for x,w in enumerate(word_mat2[i]):
            comb[w] += normprob_mat2[i][x]
        

        #print "Map", word_mat1[i][:5]
        #print "Lin", word_mat2[i][:5]
        #print "Log", word_mat3[i][:5]
        #print comb.most_common(10)
        gen_comb.append([w for (w,_) in comb.most_common(10)])

    return gen_comb

def combine_topn_maplog((word_mat1,prob_mat1),(word_mat3,prob_mat3)):

    
    prob_mat1_inv = 1-prob_mat1
    totals_1 = np.sum(prob_mat1_inv,axis=1)
    normprob_mat1 = prob_mat1_inv/totals_1[:,None]
  
   
    #print "***"
    #print prob_mat3[:2]
    totals_3 = np.sum(prob_mat3,axis=1)
    normprob_mat3 = prob_mat3/totals_3[:,None]
    #print normprob_mat3[:2]

    #for i in range(len(word_mat1)):
    gen_comb = []
    for i in range(len(word_mat1)):
    #for i in range(10):
        comb = Counter()
        for x,w in enumerate(word_mat1[i]):
            comb[w] += normprob_mat1[i][x]
        for x,w in enumerate(word_mat3[i]):
            comb[w] += normprob_mat3[i][x]

        #print "Map", word_mat1[i][:5]
        #print "Lin", word_mat2[i][:5]
        #print "Log", word_mat3[i][:5]
        #print comb.most_common(10)
        gen_comb.append([w for (w,_) in comb.most_common(10)])

    return gen_comb

def combine_topn_linlog((word_mat2,prob_mat2),(word_mat3,prob_mat3)):

    
    
    #print prob_mat2[:2]
    prob_mat2_inv = prob_mat2*-1
    #print prob_mat2_inv[:2]
    totals_2 = np.sum(prob_mat2_inv,axis=1)
    #print totals_2[:2]
    normprob_mat2 = prob_mat2_inv/totals_2[:,None]
    #print normprob_mat2[:2]

    #print "***"
    #print prob_mat3[:2]
    totals_3 = np.sum(prob_mat3,axis=1)
    normprob_mat3 = prob_mat3/totals_3[:,None]
    #print normprob_mat3[:2]

    #for i in range(len(word_mat1)):
    gen_comb = []
    for i in range(len(word_mat1)):
    #for i in range(10):
        comb = Counter()
        for x,w in enumerate(word_mat2[i]):
            comb[w] += normprob_mat2[i][x]
        for x,w in enumerate(word_mat3[i]):
            comb[w] += normprob_mat3[i][x]

        #print "Map", word_mat1[i][:5]
        #print "Lin", word_mat2[i][:5]
        #print "Log", word_mat3[i][:5]
        #print comb.most_common(10)
        gen_comb.append([w for (w,_) in comb.most_common(10)])

    return gen_comb


def trace_combine_topn_3((word_mat1,prob_mat1),(word_mat2,prob_mat2),(word_mat3,prob_mat3),testsets):


    (tst_index,test_out) = testsets[0]
    Xsaia_test = Xsaia[tst_index,:3]
    
    prob_mat1_inv = 1-prob_mat1
    totals_1 = np.sum(prob_mat1_inv,axis=1)
    normprob_mat1 = prob_mat1_inv/totals_1[:,None]
  
    #print prob_mat2[:2]
    prob_mat2_inv = prob_mat2*-1
    #print prob_mat2_inv[:2]
    totals_2 = np.sum(prob_mat2_inv,axis=1)
    #print totals_2[:2]
    normprob_mat2 = prob_mat2_inv/totals_2[:,None]
    #print normprob_mat2[:2]

    #print "***"
    #print prob_mat3[:2]
    totals_3 = np.sum(prob_mat3,axis=1)
    normprob_mat3 = prob_mat3/totals_3[:,None]
    #print normprob_mat3[:2]

    for i in range(len(word_mat1)):
    #for i in range(10):
        comb = Counter()
        for x,w in enumerate(word_mat1[i]):
            comb[w] += normprob_mat1[i][x]
        for x,w in enumerate(word_mat2[i]):
            comb[w] += normprob_mat2[i][x]
        for x,w in enumerate(word_mat3[i]):
            comb[w] += normprob_mat3[i][x]

        if comb.most_common(1)[0][0] == test_out[i]:

            if word_mat1[i][0] != test_out[i] and \
            word_mat2[i][0] != test_out[i] and \
            word_mat3[i][0] != test_out[i]:


                image_id = int(Xsaia_test[i][1])
                region_id = int(Xsaia_test[i][2])
                print "File",image_id,region_id
                bb = utils.get_saiapr_bb(image_id, region_id)
                print "bb",bb
                impath = "comb%d_%d.png"%(image_id,region_id)
                print impath
                utils.plot_labelled_bb(utils.saiapr_image_filename(image_id),[(bb  ,test_out[i])],\
                    opath=impath,omode="img")


                
                print "Map", word_mat1[i][:5]
                print "Lin", word_mat2[i][:5]
                print "Log", word_mat3[i][:5]
                print comb.most_common(10)
    

def make_results_combination():

    with gzip.open('../indata/saia_nouns159_testset.pklz', 'r') as f:
        testsets = pickle.load(f)
    

    
    print "testsets", len(testsets)
    print "Nn test regions", len(testsets[0][1])

    w2v = linwac.load_w2v()
    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in NOUNS])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodel_path = '../linmodels/linmap_nouns.pklz'
    linmodel_path = '../linmodels/linwac_vocab_w2v_nsamp74390.pklz'
    logmodel_path = '../logmodels/logwac_saia_w2v_nsamp74390.pklz'

    print "*****Eval mapmodels"
    res_map = apply_map_model(testsets,mapmodel_path,nouns_w2v_mat,NOUNS)
    print "*****Eval logmodels"
    res_log = apply_log_model(testsets,logmodel_path,NOUNS)
    print "*****Eval linmodels"
    res_lin = apply_lin_model(testsets,linmodel_path,NOUNS)

    print "Result combination map -lin -log"
    res_comb = combine_topn_maplinlog(res_map,res_lin,res_log)
    #trace_combine_topn_3(res_map,res_lin,res_log,testsets)
    get_accuracy(testsets[0][1],res_comb)

    

    print "Result combination map -lin"
    res_comb = combine_topn_maplin(res_map,res_lin)
    trace_combine_topn_3(res_map,res_lin,res_log,testsets)
    get_accuracy(testsets[0][1],res_comb)

    print "Result combination map -log"
    res_comb = combine_topn_maplog(res_map,res_log)
    #trace_combine_topn_3(res_map,res_lin,res_log,testsets)
    get_accuracy(testsets[0][1],res_comb)

    print "Result combination lin -log"
    res_comb = combine_topn_maplog(res_lin,res_log)
    #trace_combine_topn_3(res_map,res_lin,res_log,testsets)
    get_accuracy(testsets[0][1],res_comb)

def make_results_projectwac():

    with gzip.open('../indata/saia_nouns159_testset.pklz', 'r') as f:
        testsets = pickle.load(f)

    
    print "testsets", len(testsets)
    print "N test regions", len(testsets[0][1])

    w2v = linwac.load_w2v()
    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in NOUNS])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodel_path = '../linmodels/linmap_nouns.pklz'
    linmodel_path = '../linmodels/linwac_vocab_w2v_nsamp74390.pklz'
    logmodel_path = '../logmodels/logwac_saia_w2v_nsamp74390.pklz'


    print "Transfer"
    apply_map_model(testsets,mapmodel_path,nouns_w2v_mat,NOUNS)
    print "Project Wac"
    apply_project_log_model(testsets,logmodel_path,nouns_w2v_mat,NOUNS)
    print "Simple WAC2"
    apply_log_model(testsets,logmodel_path,NOUNS)
    print "Sim-WAP"
    apply_lin_model(testsets,linmodel_path,NOUNS)


def get_accuracy(gold_nouns,pred_nouns):

    acc1 = 0
    acc2 = 0
    acc5 = 0
    acc10 = 0
    total = len(gold_nouns)

    for x,lstr in enumerate(gold_nouns):
        
        l = set(lstr.split("|"))
        #print l
        #print pred_nouns[x]
        acc1 += int(len(l & set(pred_nouns[x][:1])) > 0)
        acc2 += int(len(l & set(pred_nouns[x][:2])) > 0)

        if len(pred_nouns[x]) > 2:
            acc5 += int(len(l & set(pred_nouns[x][:5])) > 0)
            acc10 += int(len(l & set(pred_nouns[x][:10])) > 0)

    print "Accuracy top1: %.2f, top2: %.2f, top5: %.2f, top10: %.2f" % \
    ((acc1/total)*100,(acc2/total)*100,(acc5/total)*100,(acc10/total)*100)    
   
    return (acc1/total,acc2/total,acc5/total,acc10/total)


if __name__ == '__main__':

    make_results_combination()
    #make_results_projectwac()