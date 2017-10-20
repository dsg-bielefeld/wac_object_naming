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
from utils import filter_by_filelist, filter_X_by_filelist, filter_by_fileregionlist
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

def eval_log_models(testsets,ttsplit,spl_models,word_mat,word_list):


    # with gzip.open('normdf_reduced_populated.pklz', 'r') as f:
    #     attdf = pickle.load(f)
    # nouns_att = {n: np.array(attdf.loc[n].tolist()) for n in NOUNS}
    # nouns_att_mat = np.array([attdf.loc[n].tolist() for n in NOUNS])
    # print "Attribute matrix", nouns_att_mat.shape
    # attpriors = np.array((attdf.sum(axis=0) / len(attdf)).tolist())

    split_accs_full10 = []
    split_accs_disjoint10 = []
    split_accs_full5 = []
    split_accs_disjoint5 = []
    split_accs_full1 = []
    split_accs_disjoint1 = []

    word_dict = {n: word_mat[x] for x,n in enumerate(word_list)}

    for tx in range(len(testsets)):
        (tst_index,tst_out) = testsets[tx]
        print "Split",tx
        Xsaia_test = Xsaia[tst_index,3:]
        print "Nouns",ttsplit[tx]['nouns']
        
        #nouns_att_mat_tst = np.array([attdf.loc[n].tolist() for n in ttsplit[x]['nouns']])

        with gzip.open(spl_models[tx], 'r') as f:
            spl_wac = pickle.load(f)
        for n in ttsplit[tx]['nouns']:
            print n, spl_wac[n]

        projected_words_full10 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,word_list,10,10)
        projected_words_disjoint10 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,ttsplit[tx]['nouns'],10,2)

        print "Eval on full set of nouns"
        split_accs_full10.append(get_accuracy(tst_out,projected_words_full10))
        print "Eval on disjoint set of nouns"
        split_accs_disjoint10.append(get_accuracy(tst_out,projected_words_disjoint10))

        print "Projecting 5 into distributional space"
        
        projected_words_full5 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,word_list,5,10)
        projected_words_disjoint5 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,ttsplit[tx]['nouns'],5,2)


        print "Eval on full set of nouns"
        split_accs_full5.append(get_accuracy(tst_out,projected_words_full5))
        print "Eval on disjoint set of nouns"
        split_accs_disjoint5.append(get_accuracy(tst_out,projected_words_disjoint5))

        print "Projecting 1 into distributional space"
        
        projected_words_full1 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,word_list,1,10)
        projected_words_disjoint1 = decode_and_map_wac(Xsaia_test,spl_wac,word_mat,word_list,ttsplit[tx]['nouns'],1,2)

        print "Eval on full set of nouns"
        split_accs_full1.append(get_accuracy(tst_out,projected_words_full1))
        print "Eval on disjoint set of nouns"
        split_accs_disjoint1.append(get_accuracy(tst_out,projected_words_disjoint1))

        # print "Projecting into attribute space"
        # pred_att_embeddings = words2embedding_weighted(pred_words,pred_probs,nouns_att)
        # #projected_att_words_full,_ = get_nearest_signature(pred_att_embeddings,nouns_att_mat,attpriors,NOUNS,10)
        # projected_att_words_full,_ = get_nearest_neighbours(pred_att_embeddings,nouns_att_mat,NOUNS,10)
        # #projected_att_words_disjoint,_ = get_nearest_signature(pred_att_embeddings,nouns_att_mat_tst,attpriors,ttsplit[x]['nouns'],2)
        # projected_att_words_disjoint,_ = get_nearest_neighbours(pred_att_embeddings,nouns_att_mat_tst,ttsplit[x]['nouns'],2)


        # print "Eval on full set of nouns"
        # split_accs_full_att.append(get_accuracy(tst_out,projected_att_words_full))
        # print "Eval on disjoint set of nouns"
        # split_accs_disjoint_att.append(get_accuracy(tst_out,projected_att_words_disjoint))

    t = len(split_accs_full10)
    acc1 = sum([x for (x,_,_,_) in split_accs_full10])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full10])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full10])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full10])/t

    print "\n****Full testsets"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)

    results = [('full set','wac, project top10',acc1,acc2,acc5,acc10)]

    

    t = len(split_accs_disjoint10)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint10])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint10])/t
    print "\n****Disjoint testsets"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    results.append(('disjoint','wac, project top10',acc1,acc2,0,0))

    t = len(split_accs_full5)
    acc1 = sum([x for (x,_,_,_) in split_accs_full5])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full5])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full5])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full5])/t

    print "\n****Full testsets"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)

    results.append(('full set','wac, project top5',acc1,acc2,acc5,acc10))

    t = len(split_accs_disjoint5)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint5])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint5])/t
    print "\n****Disjoint testsets"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    results.append(('disjoint','wac, project top5',acc1,acc2,0,0))


    t = len(split_accs_full1)
    acc1 = sum([x for (x,_,_,_) in split_accs_full1])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full1])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full1])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full1])/t

    print "\n****Full testsets"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)

    results.append(('full set','wac, project top1',acc1,acc2,acc5,acc10))

    t = len(split_accs_disjoint1)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint1])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint1])/t
    print "\n****Disjoint testsets"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    results.append(('disjoint','wac, project top1',acc1,acc2,0,0))

    # print "\n****Full testsets, attributional"
    # print "Average acc1", sum([x for (x,_,_,_) in split_accs_full_att])/len(split_accs_full_att)
    # print "Average acc2", sum([x for (_,x,_,_) in split_accs_full_att])/len(split_accs_full_att)
    # print "Average acc5", sum([x for (_,_,x,_) in split_accs_full_att])/len(split_accs_full_att)
    # print "Average acc10", sum([x for (_,_,_,x) in split_accs_full_att])/len(split_accs_full_att)

    # print "\n****Disjoint testsets,attributional"
    # print "Average acc1", sum([x for (x,_,_,_) in split_accs_disjoint_att])/len(split_accs_disjoint_att)
    # print "Average acc2", sum([x for (_,x,_,_) in split_accs_disjoint_att])/len(split_accs_disjoint_att)

    return results


def eval_map_models(testsets,ttsplit,spl_models,word_mat,word_list):

    word_dict = {n: word_mat[x] for x,n in enumerate(word_list)}

    split_accs_full = []
    split_accs_disjoint = []
    for tx in range(len(testsets)):
        (tst_index,tst_out) = testsets[tx]
        print "Split",tx
        Xsaia_test = Xsaia[tst_index,3:]
        print "Nouns",ttsplit[tx]['nouns']
        word_mat_tst = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in ttsplit[tx]['nouns']])

        with gzip.open(spl_models[tx], 'r') as f:
            spl_map = pickle.load(f)

        projected_words_full = decode_transfer(Xsaia_test,spl_map,word_mat,word_list,10)
        projected_words_disjoint = decode_transfer(Xsaia_test,spl_map,word_mat_tst,ttsplit[tx]['nouns'],2)

        print "Eval on full set of nouns"
        split_accs_full.append(get_accuracy(tst_out,projected_words_full))
        print "Eval on disjoint set of nouns"
        split_accs_disjoint.append(get_accuracy(tst_out,projected_words_disjoint))



    t = len(split_accs_full)
    acc1 = sum([x for (x,_,_,_) in split_accs_full])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full])/t

    print "\n****Full testsets"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)
    
    results = [('full set','transfer',acc1,acc2,acc5,acc10)]

    t = len(split_accs_disjoint)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint])/t
    print "\n****Disjoint testsets"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    
    results.append(('disjoint','transfer',acc1,acc2,0,0))

    return results


def eval_lin_models(testsets,ttsplit,spl_models,word_mat,word_list):

    
    split_accs_full = []
    split_accs_disjoint = []
    split_accs_full_proj = []
    split_accs_disjoint_proj = []

    for tx in range(len(testsets)):
        (tst_index,tst_out) = testsets[tx]
        print "Split",tx
        Xsaia_test = Xsaia[tst_index,3:]
        print "Nouns",ttsplit[tx]['nouns']

        with gzip.open(spl_models[tx], 'r') as f:
            spl_wac = pickle.load(f)
        spl_wac_disjoint = {n:spl_wac[n] for n in ttsplit[tx]['nouns'] }

        pred_words_full,pred_probs_full = get_topn_linwacs(Xsaia_test,spl_wac,10)
        pred_words_disjoint,pred_probs_disjoint = get_topn_linwacs(Xsaia_test,spl_wac_disjoint,5)

        print "Projecting into distributional space"
        projected_words_full = decode_and_map_linwap(Xsaia_test,spl_wac,word_mat,word_list,word_list,5,10)
        projected_words_disjoint = decode_and_map_linwap(Xsaia_test,spl_wac,word_mat,word_list,ttsplit[tx]['nouns'],5,2)

        print "Eval on full set of nouns"
        split_accs_full.append(get_accuracy(tst_out,pred_words_full))
        print "Eval on disjoint set of nouns"
        split_accs_disjoint.append(get_accuracy(tst_out,pred_words_disjoint))

        print "Eval on full set of nouns, projected"
        split_accs_full_proj.append(get_accuracy(tst_out,projected_words_full))
        print "Eval on disjoint set of nouns,projected"
        split_accs_disjoint_proj.append(get_accuracy(tst_out,projected_words_disjoint))

    t = len(split_accs_full)
    acc1 = sum([x for (x,_,_,_) in split_accs_full])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full])/t

    results = [('full set','sim-wap',acc1,acc2,acc5,acc10)]

    print "\n****Full testsets, predictions"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)
    t = len(split_accs_disjoint)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint])/t
    print "\n****Disjoint testsets, predictions"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    results.append(('disjoint','sim-wap',acc1,acc2,0,0))



    t = len(split_accs_full_proj)
    acc1 = sum([x for (x,_,_,_) in split_accs_full_proj])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_full_proj])/t
    acc5 = sum([x for (_,_,x,_) in split_accs_full_proj])/t
    acc10 = sum([x for (_,_,_,x) in split_accs_full_proj])/t

    results.append(('full set','sim-wap, project top5',acc1,acc2,acc5,acc10))

    print "\n****Full testsets, predictions"
    print "Average acc1: %.2f, acc2: %.2f, acc5: %.2f, acc10: %.2f" % (acc1,acc2,acc5,acc10)
    t = len(split_accs_disjoint)
    acc1 = sum([x for (x,_,_,_) in split_accs_disjoint_proj])/t
    acc2 = sum([x for (_,x,_,_) in split_accs_disjoint_proj])/t
    print "\n****Disjoint testsets, predictions"
    print "Average acc1: %.2f, acc2: %.2f" % (acc1,acc2)

    results.append(('disjoint','sim-wap, project top5',acc1,acc2,0,0))

    return results



def get_accuracy(gold_nouns,pred_nouns):

    acc1 = 0
    acc2 = 0
    acc5 = 0
    acc10 = 0
    total = len(gold_nouns)

    for x,lstr in enumerate(gold_nouns):
        #print l
        #print pred_nouns[x]
        l = set(lstr.split("|"))
        acc1 += int(len(l & set(pred_nouns[x][:1])) > 0)
        acc2 += int(len(l & set(pred_nouns[x][:2])) > 0)

        if len(pred_nouns[x]) > 2:
            acc5 += int(len(l & set(pred_nouns[x][:5])) > 0)
            acc10 += int(len(l & set(pred_nouns[x][:10])) > 0)

    print "Accuracy top1: %.2f, top2: %.2f, top5: %.2f, top10: %.2f" % \
    (acc1/total,acc2/total,acc5/total,acc10/total)    
   
    return (acc1/total,acc2/total,acc5/total,acc10/total)


def make_results_randomsplits():

    with gzip.open('../indata/saia_zeroshot_nounsplits_testsets.pklz', 'r') as f:
        testsets = pickle.load(f)
    with open('../indata/saia_zeroshot_nounsplits.json', 'r') as f:
        ttsplit = json.load(f)

    
    print "testsets", len(testsets)

    w2v = linwac.load_w2v()
    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in NOUNS])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodels = ['../linmodels/linmap_nouns_zeroshot_split'+str(x)+'.pklz' for x in range(10)]
    linmodels = ['../linmodels/linwac_nouns_w2v_zeroshot_split'+str(x)+'.pklz' for x in range(10)]
    logmodels = ['../logmodels/logwac_saia_nouns_zeroshot_split'+str(x)+'_nosamp.pklz' for x in range(10)]

    print "*****Eval mapmodels"
    res1 = eval_map_models(testsets[:10],ttsplit[:10],mapmodels,nouns_w2v_mat,NOUNS)
    print "*****Eval logmodels"
    res2 = eval_log_models(testsets[:10],ttsplit[:10],logmodels,nouns_w2v_mat,NOUNS)
    print "*****Eval linmodels"
    res3 = eval_lin_models(testsets[:10],ttsplit[:10],linmodels,nouns_w2v_mat,NOUNS)

    results = []
    for (x,y,a,b,c,d) in res1+res2+res3:
        results.append((x,y,"%.2f"%(a*100),"%.2f"%(b*100),"%.2f"%(c*100),"%.2f"%(d*100)))

    df = pd.DataFrame(results,columns=['testset','model','@1','@2','@5','@10'])

    print df.to_latex(index=False)


def make_results_hypernyms():

    with gzip.open('../indata/saia_zeroshot_hypernsplit_testset.pklz', 'r') as f:
        testsets = pickle.load(f)
    with open('../indata/saia_zeroshot_hypernsplit.json', 'r') as f:
        ttsplit_dict = json.load(f)

    this_wordlist = NOUNS + [n for n in ttsplit_dict['nouns'] if not n in NOUNS]

    print "Train", len(ttsplit_dict['train'])
    print "Test", len(ttsplit_dict['test'])

    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in this_wordlist])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodels = ['../linmodels/linmap_nouns_zeroshot_hypernsplit.pklz']
    linmodels = ['../linmodels/linwac_nouns__zeroshot_hypernsplit.pklz']
    logmodels = ['../logmodels/logwac_saia_nouns_zeroshot_hypernsplit_nosamp.pklz']

    print "*****Eval mapmodels"
    res1 = eval_map_models(testsets,[ttsplit_dict],mapmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval logmodels"
    res2 = eval_log_models(testsets,[ttsplit_dict],logmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval linmodels"
    res3 = eval_lin_models(testsets,[ttsplit_dict],linmodels,nouns_w2v_mat,this_wordlist)

    results = []
    for (x,y,a,b,c,d) in res1+res2+res3:
        results.append((x,y,"%.2f"%(a*100),"%.2f"%(b*100),"%.2f"%(c*100),"%.2f"%(d*100)))

    df = pd.DataFrame(results,columns=['testset','model','@1','@2','@5','@10'])

    print df.to_latex(index=False)


def make_results_plurals():

    with gzip.open('../indata/saia_zeroshot_pluralsplit_testset.pklz', 'r') as f:
        testsets = pickle.load(f)
    with open('../indata/saia_zeroshot_pluralsplit.json', 'r') as f:
        tt_dict = json.load(f)

    print "Train", len(tt_dict['train'])
    print "Test", len(tt_dict['test'])

    this_wordlist = tt_dict['singulars'] + tt_dict['nouns']

    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in this_wordlist])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodels = ['../linmodels/linmap_nouns_zeroshot_pluralsplit.pklz']
    linmodels = ['../linmodels/linwac_nouns__zeroshot_pluralsplit.pklz']
    logmodels = ['../logmodels/logwac_saia_nouns_zeroshot_pluralsplit_nosamp.pklz']

    print "*****Eval mapmodels"
    res1 = eval_map_models(testsets,[tt_dict],mapmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval logmodels"
    res2 = eval_log_models(testsets,[tt_dict],logmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval linmodels"
    res3 = eval_lin_models(testsets,[tt_dict],linmodels,nouns_w2v_mat,this_wordlist)

    results = []
    for (x,y,a,b,c,d) in res1+res2+res3:
        results.append((x,y,"%.2f"%(a*100),"%.2f"%(b*100),"%.2f"%(c*100),"%.2f"%(d*100)))

    df = pd.DataFrame(results,columns=['testset','model','@1','@2','@5','@10'])

    print df.to_latex(index=False)


def make_results_mixedplurals():

    with gzip.open('../indata/saia_zeroshot_mixedpluralsplit_testset.pklz', 'r') as f:
        testsets = pickle.load(f)
    with open('../indata/saia_zeroshot_mixedpluralsplit.json', 'r') as f:
        tt_dict = json.load(f)

    print "Train", len(tt_dict['train'])
    print "Test", len(tt_dict['test'])

    this_wordlist = tt_dict['singulars'] + tt_dict['nouns']

    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in this_wordlist])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodels = ['../linmodels/linmap_nouns_zeroshot_mixedpluralsplit.pklz']
    linmodels = ['../linmodels/linwac_nouns__zeroshot_mixedpluralsplit.pklz']
    logmodels = ['../logmodels/logwac_saia_nouns_zeroshot_mixedpluralsplit_nosamp.pklz']

    print "*****Eval mapmodels"
    res1 = eval_map_models(testsets,[tt_dict],mapmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval logmodels"
    res2 = eval_log_models(testsets,[tt_dict],logmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval linmodels"
    res3 = eval_lin_models(testsets,[tt_dict],linmodels,nouns_w2v_mat,this_wordlist)

    results = []
    for (x,y,a,b,c,d) in res1+res2+res3:
        results.append((x,y,"%.2f"%(a*100),"%.2f"%(b*100),"%.2f"%(c*100),"%.2f"%(d*100)))

    df = pd.DataFrame(results,columns=['testset','model','@1','@2','@5','@10'])

    print df.to_latex(index=False)

def make_results_standard_plurals():

    with gzip.open('../indata/saia_standard_pluralsplit_testset.pklz', 'r') as f:
        testsets = pickle.load(f)
    with open('../indata/saia_standard_pluralsplit.json', 'r') as f:
        tt_dict = json.load(f)

    this_wordlist = tt_dict['nouns']

    nouns_w2v_mat = np.array([msim.w2v_vecs[msim.word2ind[n]] for n in this_wordlist])
    print "W2v matrix", nouns_w2v_mat.shape

    mapmodels = ['../linmodels/linmap_nouns_standard_pluralsplit.pklz']
    linmodels = ['../linmodels/linwac_nouns__standard_pluralsplit.pklz']
    logmodels = ['../logmodels/logwac_saia_nouns_standard_pluralsplit_nosamp.pklz']

    print "*****Eval mapmodels"
    res1 = eval_map_models(testsets,[tt_dict],mapmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval logmodels"
    res2 = eval_log_models(testsets,[tt_dict],logmodels,nouns_w2v_mat,this_wordlist)
    print "*****Eval linmodels"
    res3 = eval_lin_models(testsets,[tt_dict],linmodels,nouns_w2v_mat,this_wordlist)

    results = []
    for (x,y,a,b,c,d) in res1+res2+res3:
        results.append((x,y,"%.2f"%(a*100),"%.2f"%(b*100),"%.2f"%(c*100),"%.2f"%(d*100)))

    df = pd.DataFrame(results,columns=['testset','model','@1','@2','@5','@10'])

    print df.to_latex(index=False)



if __name__ == '__main__':

    make_results_randomsplits()
    make_results_hypernyms()
    make_results_plurals()
    #make_results_standard_plurals()
    #make_results_mixedplurals()

    
