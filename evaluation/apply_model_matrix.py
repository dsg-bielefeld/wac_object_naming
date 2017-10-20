import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist, cdist



NOUNS = [l.strip() for l in open('noun_list.txt').readlines()]

def decode_transfer(Xtest,mapmodel,word_mat,word_list,topn):

    pred_embeddings = get_transfer_embeddings(Xtest,mapmodel)
    projected_words,_ = get_nearest_neighbours(pred_embeddings,word_mat,word_list,topn)

    return projected_words

def decode_transfer_prob(Xtest,mapmodel,word_mat,word_list,topn):

    pred_embeddings = get_transfer_embeddings(Xtest,mapmodel)
    projected_words,word_probs = get_nearest_neighbours(pred_embeddings,word_mat,word_list,topn)

    return projected_words,word_probs

def decode_and_map_wac(Xtest,wacmodel,word_mat,word_list,tst_word_list,topn_proj,topn_out):

    word_dict = {n: word_mat[x] for x,n in enumerate(word_list)}

    pred_words,pred_probs = get_topn_logwacs(Xtest,wacmodel,topn_proj)
    pred_embeddings = words2embedding_weighted(pred_words,pred_probs,word_dict)

    word_mat_tst = np.array([word_mat[word_list.index(n)] for n in tst_word_list])
    projected_words,_ = get_nearest_neighbours(pred_embeddings,word_mat_tst,tst_word_list,topn_out)

    return projected_words

def decode_and_map_linwap(Xtest,linmodel,word_mat,word_list,tst_word_list,topn_proj,topn_out):

    word_dict = {n: word_mat[x] for x,n in enumerate(word_list)}

    pred_words,pred_probs = get_topn_linwacs(Xtest,linmodel,topn_proj)
    pred_embeddings = words2embedding_weighted(pred_words,pred_probs,word_dict)

    word_mat_tst = np.array([word_mat[word_list.index(n)] for n in tst_word_list])
    projected_words,_ = get_nearest_neighbours(pred_embeddings,word_mat_tst,tst_word_list,topn_out)

    return projected_words

def get_topn_logwacs(X,logmodel,topn,nounlist=NOUNS):

    #print "Test set", X.shape
    
    word_columns = []
    for n in nounlist:
        probcol = np.zeros(X.shape[0])
        if n in logmodel:
            if logmodel[n]:
                probcol = logmodel[n].predict_proba(X)[:,1]
        word_columns.append(1-probcol)
  
    word_matrix = np.column_stack(word_columns)
    #print "Stacked word predictions", word_matrix.shape

    word_sort = np.argsort(word_matrix,axis=1)
    #print "Argsort on word columns",word_sort.shape

    word_sort = word_sort[:,:topn]
    #print "Topn predictions",word_sort.shape
    #print word_sort[:10]

    word_predictions = []
    for i in range(word_sort.shape[0]):
        word_predictions.append([nounlist[x] for x in word_sort[i]])
    #     if i < 10:
    #         print [nounlist[x] for x in word_sort[i]]
    # print "****"
    word_probs = []
    for i in range(word_sort.shape[0]):
        word_probs.append([1-word_matrix[i][x] for x in word_sort[i]])
        # if i <10:
        #     print [1-word_matrix[i][x] for x in word_sort[i]] 
    
    return word_predictions,np.array(word_probs)


def get_topn_linwacs(X,linmodel,topn,nounlist=NOUNS):

    #print "Test set", X.shape
    
    word_columns = []
    for n in nounlist:
        probcol = np.zeros(X.shape[0])+1
        if n in linmodel:
            if linmodel[n]:
                probcol = linmodel[n].predict(X)
        word_columns.append(probcol)
  
    word_matrix = np.column_stack(word_columns)
    #print "Stacked word predictions", word_matrix.shape

    word_sort = np.argsort(word_matrix,axis=1)
    #print "Argsort on word columns",word_sort.shape

    word_sort = word_sort[:,:topn]
    #print "Topn predictions",word_sort.shape
    #print word_sort[:10]

    word_predictions = []
    for i in range(word_sort.shape[0]):
        word_predictions.append([nounlist[x] for x in word_sort[i]])
    #     if i < 10:
    #         print [nounlist[x] for x in word_sort[i]]
    # print "****"
    word_probs = []
    for i in range(word_sort.shape[0]):
        word_probs.append([word_matrix[i][x] for x in word_sort[i]])
        # if i <10:
        #     print [1-word_matrix[i][x] for x in word_sort[i]] 
    
    return word_predictions,np.array(word_probs)

def get_transfer_embeddings(X,mapmodel):

    #print "Test set", X.shape
    
    vec_matrix =  np.column_stack([reg.predict(X) for reg in mapmodel])
    #print "Stacked word predictions", vec_matrix.shape
    
    return vec_matrix

def words2embedding_weighted(word_predictions,word_probs,word_vectors):

    vecs = []
    for i in range(len(word_predictions)):
            wlist = word_predictions[i]
            plist = word_probs[i]
            new_vec = np.sum([word_vectors[w].astype(float)*plist[x] for x,w in enumerate(wlist)],axis=0)
            vecs.append(new_vec)
    
    vecs = np.array(vecs)
    #print "Embedding matrix",vecs.shape

    return vecs


def get_nearest_neighbours(vec_predicted,vec_words,wordlist,topn):

    dist_matrix = cdist(vec_predicted,vec_words,'cosine')
    #print dist_matrix.shape

    word_sort = np.argsort(dist_matrix,axis=1)
    #print "Argsort on word columns",word_sort.shape

    word_sort = word_sort[:,:topn]
    #print "Topn predictions",word_sort.shape
    #print word_sort[:10]

    word_predictions = []
    for i in range(word_sort.shape[0]):
        word_predictions.append([wordlist[x] for x in word_sort[i]])
    #     if i < 10:
    #         print [wordlist[x] for x in word_sort[i]]
    # print "****"
    word_probs = []
    for i in range(word_sort.shape[0]):
        word_probs.append([dist_matrix[i][x] for x in word_sort[i]])
        # if i <10:
        #     print [dist_matrix[i][x] for x in word_sort[i]]

    return word_predictions,np.array(word_probs)

def get_nearest_signature(vec_predicted,vec_words,word_priors,wordlist,topn):
    

    word_scores = []
    for x in range(vec_words.shape[0]):
        this_signature = vec_words[x]
        # This is formula 2 of Lampert et al. 2009, in the formulation of
        #   Rohrbach et al. 2010 (i.e., using power to set
        #   0 attributes = 1, and then taking product), also formula 2.
        word_scores.append(np.power((vec_predicted / word_priors), 
                                     this_signature).prod(axis=1)*-1) # mult by -1 just for sorting purposes! 

    word_scores = np.column_stack(word_scores)
    #print "Signature score matirx",word_scores.shape

    word_sort = np.argsort(word_scores,axis=1)
    #print "Argsort on word columns",word_sort.shape

    word_sort = word_sort[:,:topn]
    #print "Topn predictions",word_sort.shape

    word_predictions = []
    for i in range(word_sort.shape[0]):
        word_predictions.append([wordlist[x] for x in word_sort[i]])

    word_probs = []
    for i in range(word_sort.shape[0]):
        word_probs.append([word_scores[i][x] for x in word_sort[i]])

    return word_predictions,word_probs