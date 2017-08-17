# mymod.py
"""Python module demonstrates passing MATLAB types to Python functions"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

"""
Determinantal point process sampling procedures
"""
def search(words):
    """Return list of words containing 'son'"""
    newlist = [w for w in words if 'son' in w]
    return newlist

def draw_prob(X, Y, num):
    """
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    fig = plt.figure()
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    
    ax.set_title('sampling probability at different level')    
    ax.set_xlabel('data index')
    ax.set_ylabel('sampling probability')
    
    for i in range(num):
        bios = len(X[i])-len(Y[i])
        ax.plot(X[i][bios:], Y[i][0:], colors[i % len(colors)], label='level-%d' % i)
    plt.xlim(0,200)
    plt.hlines([0.5], 0, 200, linestyles="dashed")  # y=0.5??????????????????
    plt.legend()
    plt.show()

def build_cosine_similary_matrix(items):
    """
    build the cosine similarity matrix 
    """
    L = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(i, len(items)):
            L[i, j] = np.inner(items[i], items[j])/(np.linalg.norm(items[i])*np.linalg.norm(items[j]))
            L[j, i] = L[i, j]
    return L    
    
def build_similary_matrix(cov_function, items):
    """
    build the similarity matrix from a covariance function
    cov_function and a set of items. each pair of items
    is given to cov_function, which computes the similarity
    between two items.
    """
    L = np.zeros((len(items), len(items)))
    for i in range(len(items)):
        for j in range(i, len(items)):
            L[i, j] = cov_function(items[i], items[j])
            L[j, i] = L[i, j]
    return L

def exp_quadratic(sigma):
    """
    exponential quadratic covariance function
    """
    def f(p1, p2):
        return np.exp(-0.5 * (((p1 - p2)**2).sum()) / sigma**2)
    return f

class DPP:
    def __init__(self, sigma = 3.0, K = 200, step = 2, start_size = 200):
        self.sigma = sigma
        self.K = K
        self.step = step
        self.start_size = start_size
        
    def sample_standard(self, items, L, max_nb_iterations=1000, rng=np.random):
        """
        Sample a list of items from a DPP defined
        by the similarity matrix L. The algorithm
        is iterative and not using fast DPP.
        """
        initial = rng.choice(range(len(items)), size=self.start_size, replace=False)
        Y = [False] * len(items)
        for i in initial:
            Y[i] = True
        Y = np.array(Y)
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]
        L_Y_inv = np.linalg.inv(L_Y)
        for i in range(max_nb_iterations):
            u = rng.randint(0, len(items))
            # insertion transition        
            if Y[u] == False:
                c_u = L[u:u+1, :]
                c_u = c_u[:, u:u+1]
                b_u = L[Y, :]
                b_u = b_u[:, u:u+1]
                p_include_U = min(1, c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                #print "include probability: %.4f" % p_include_U
                if rng.uniform() <= p_include_U:                
                    Y[u] = True
                    L_Y = L[Y, :]
                    L_Y = L_Y[:, Y]
                    L_Y_inv = np.linalg.inv(L_Y)
            else:
                X = Y.copy()
                X[u] = False
                L_X = L[X, :]
                L_X = L_X[:, X]
                L_X_inv = np.linalg.inv(L_X)
                c_u = L[u:u+1, :]
                c_u = c_u[:, u:u+1]
                b_u = L[X, :]
                b_u = b_u[:, u:u+1]
                p_remove_U = min(1, 1 - (c_u - np.dot(np.dot(b_u.T, L_X_inv), b_u)))
                #print "remove probability: %.4f" % (p_remove_U)
                if rng.uniform() <= p_remove_U:
                    Y[u] = False
                    L_Y_inv = L_X_inv
                    
        sample = []
        for i in range(len(items)):
            if Y[i] == True:
                sample.append(i)
        print "%d samples selected from [%d] ground set [sigma:%.2f,start size:%d]" % (len(sample), len(L), self.sigma, self.start_size)
        return sample

    def sample_fast(self, items, L, max_nb_iterations=1000, rng=np.random):
        """
        Sample a list of items from a DPP defined
        by the similarity matrix L. The algorithm
        is iterative and runs for max_nb_iterations.
        The algorithm used is from
        (Fast Determinantal Point Process Sampling with
        Application to Clustering, Byungkon Kang, NIPS 2013)
        !!has problem for updating L_Y_inv!!
        """
        #Y = rng.choice((True, False), size=len(items))
        initial = rng.choice(range(len(items)), size=self.start_size, replace=False)
        Y = [False] * len(items)
        for i in initial:
            Y[i] = True
        Y = np.array(Y)
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]
        L_Y_inv = np.linalg.inv(L_Y)
    
        for i in range(max_nb_iterations):
            u = rng.randint(0, len(items))
            # insertion transition        
            if Y[u] == False:
                c_u = L[u:u+1, :]
                c_u = c_u[:, u:u+1]
                b_u = L[Y, :]
                b_u = b_u[:, u:u+1]
                p_include_U = min(1, c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                #print "include probability: %.4f" % p_include_U
                if rng.uniform() <= p_include_U:
                    d_u = (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                    upleft = (L_Y_inv +
                              np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T),
                                     L_Y_inv) / d_u)
                    upright = -np.dot(L_Y_inv, b_u) / d_u
                    downleft = -np.dot(b_u.T, L_Y_inv) / d_u
                    downright = d_u
                    L_Y_inv = np.bmat([[upleft, upright], [downleft, downright]])
                    Y[u] = True
                    L_Y = L[Y, :]
                    L_Y = L_Y[:, Y]
                    #L_Y_inv = np.linalg.inv(L_Y)
            # deletion transition
            else:
                X = Y
                X[u] = False
                L_X = L[X, :]
                L_X = L_X[:, X]
                L_X_inv = np.linalg.inv(L_X)
                c_u = L[u:u+1, :]
                c_u = c_u[:, u:u+1]
                b_u = L[X, :]
                b_u = b_u[:, u:u+1]
    #            l = L_Y_inv.shape[0] - 1
    #            D = L_Y_inv[0:l, :]
    #            D = D[:, 0:l]
    #            e = L_Y_inv[0:l, :]
    #            e = e[:, l:l+1]
    #            f = L_Y_inv[l:l+1, :]
    #            f = f[:, l:l+1]
    #            L_Yu_inv = D - np.dot(e, e.T) / f
                p_remove_U = min(1, 1./(0.0001+ c_u - np.dot(np.dot(b_u.T, L_X_inv), b_u)))
                #print "remove probability: %.4f" % p_remove_U
                if rng.uniform() <= p_remove_U:                
                    Y[u] = False
                    L_Y = L[Y, :]
                    L_Y = L_Y[:, Y]
                    L_Y_inv = L_X_inv
                    #L_Y_inv = np.linalg.inv(L_Y)
        #return np.array(items)[Y]
        sample = []
        for i in range(len(items)):
            if Y[i] == True:
                sample.append(i)
        print "%d samples selected from [%d] ground set [sigma:%.2f,start size:%d]" % (len(sample), len(L), self.sigma, self.start_size)
        return sample
        
    def sample_fast_stream(self, feats, rng=np.random):
        """
        Sample a list of items from a DPP defined by the similarity matrix L. 
        The algorithm used is from (Fast Determinantal Point Process Sampling 
        with Application to Clustering, Byungkon Kang, NIPS 2013).
        The algorithm is adapted to streaming setting with only insertion.
        """
        Y = [0]
        L = build_cosine_similary_matrix(feats[Y,:])
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]
        L_Y_inv = np.linalg.inv(L_Y)
        min_prob = 1.0
        max_prob = 0.0
        for i in range(1, len(feats), self.step):
            u = i
            L = build_cosine_similary_matrix(feats[Y+[u],:])
            c_u = L[len(L)-1:len(L), :]
            c_u = c_u[:, len(L)-1:len(L)]
            b_u = L[0:len(L)-1, :]
            b_u = b_u[:, len(L)-1:len(L)]
    
            p_include_U = c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)
            if p_include_U < min_prob:
                min_prob = p_include_U
            if p_include_U > max_prob:
                max_prob = p_include_U
            if rng.uniform() <= p_include_U:
                Y.append(u)
                d_u = (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u))
                upleft = (L_Y_inv +
                          np.dot(np.dot(np.dot(L_Y_inv, b_u), b_u.T),
                                 L_Y_inv) / d_u)
                upright = -np.dot(L_Y_inv, b_u) / d_u
                downleft = -np.dot(b_u.T, L_Y_inv) / d_u
                downright = d_u
                L_Y_inv = np.bmat([[upleft, upright], [downleft, downright]])
                #print "add one sample with probability: %.4f" % p_include_U
                
        sample = Y
        sample.sort()
        print "fast streamDPP: %d samples from [%d] ground set prob:[%.4f - %.4f]" % (len(sample), len(feats), min_prob, max_prob)
        return sample
        
    def sample_stream(self, feats, rng=np.random):
        """
        Sample a list of items from a DPP defined by the similarity matrix L. 
        The algorithm is adapted to streaming setting with only insertion.
        """
        Y = [0]
        L = build_cosine_similary_matrix(feats[Y,:])
        L_Y = L[Y, :]
        L_Y = L_Y[:, Y]
        L_Y_inv = np.linalg.inv(L_Y)
        min_prob = 1.0
        max_prob = 0.0
        for i in range(1, len(feats), self.step):
            u = i
            L = build_cosine_similary_matrix(feats[Y+[u],:])
            L_Y = L[0:len(L)-1,:]
            L_Y = L_Y[:,0:len(L)-1]
            L_Y_inv = np.linalg.inv(L_Y)
            c_u = L[len(L)-1:len(L), :]
            c_u = c_u[:, len(L)-1:len(L)]            
            b_u = L[0:len(L)-1, :]
            b_u = b_u[:, len(L)-1:len(L)]
    
            p_include_U = c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)
            if p_include_U < min_prob:
                min_prob = p_include_U
            if p_include_U > max_prob:
                max_prob = p_include_U
            if rng.uniform() <= p_include_U:
                Y.append(u)
                #print "add one sample with probability: %.4f" % p_include_U
                
        sample = Y
        sample.sort()
        print "streamDPP: %d samples from [%d] ground set prob:[%.4f - %.4f]" % (len(sample), len(feats), min_prob, max_prob)
        return sample
        
    def sample_hierarchy_stream(self, feats, rng=np.random):
        """
        Sample a hierarchical list of items from a DPP defined by the similarity matrix L. 
        The algorithm is adapted to streaming setting with only insertion.
        """       
        Level = 1
        samples = [[0]]
        probs = [[1.0]]
        for i in range(1, len(feats)):
            l = 0
            while (l<Level):
                L = build_cosine_similary_matrix(feats[samples[l]+[i],:])
                L_Y = L[0:len(L)-1,:]
                L_Y = L_Y[:,0:len(L)-1]
                L_Y_inv = np.linalg.inv(L_Y)
                c_u = L[len(L)-1, len(L)-1]
                b_u = L[0:len(L)-1, :]
                b_u = b_u[:, -1]
        
                p_include = min(1, (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
                probs[l].append(p_include)
                if rng.uniform() <= p_include:
                    #print "level %d: add one sample with probability: %.4f" % (l, p_include)
                    samples[l].append(i)
                    l += 1
                else:
                    break
            if l == Level:
                Level += 1
                samples.append([i])
                probs.append([1.0])
        for i in range(Level):
            if len(samples[i])<2:
                break
            print "streamDPP (level-%d): %d samples from [%d] ground set, prob:[%.4f-%.4f]"\
                % (i, len(samples[i]), len(feats), min(probs[i][1:]), max(probs[i][1:]))
        #draw_prob([range(len(feats))]+samples, probs, 1)
        #return
        
        # get relations between adjacent levels
        edges = []
        for i in range(1, len(samples)):
            parent = samples[-i]
            child = samples[-i-1]
            L_sub = np.zeros((len(parent),len(child)), np.float32)
            # constrain the clustering to be temporally local
            for m in range(len(parent)):
                for n in range(len(child)):
                    #L_sub[m,n] = np.exp(-0.5 * (((feats[parent[m],:] - feats[child[n],:])**2).sum()) / self.sigma**2)
                    L_sub[m,n] = np.inner(feats[parent[m],:], feats[child[n],:])/(np.linalg.norm(feats[parent[m],:])*np.linalg.norm(feats[child[n],:]))
                    # constrain the clustering to be temporally local
                    # lower-bound child
#                    if m-1 >= 0 and child[n] <= parent[m-1]:
#                        L_sub[m,n] = 0
                    # upper-bound child
#                    if m+1 < len(parent) and child[n] >= parent[m+1]:
#                        L_sub[m,n] = 0
            idx = np.argmax(L_sub, axis=0)
            edge = {}
            for j in range(len(idx)):
                if edge.has_key(parent[idx[j]]):
                    edge[parent[idx[j]]].append(child[j])
                else:
                    edge[parent[idx[j]]] = [child[j]]
            edges.append(edge)                
        
        return edges
    
    def sample_seq(self, feats, rng=np.random):
        """
        Sequentially sample a list of items considering both diversity and quality. 
        The algorithm used is adapted from (Diverse Sequential Subset Selection
        for Supervised Video Summarization, NIPS 2014).
        """
        sample = [0]
        min_prob = 1.0
        max_prob = 0.0
        for i in range(1, len(feats), self.step):
            node_feat = feats[i, :]
            prev_feat = feats[sample[-1],:]
            #sim = np.exp(-0.5 * (((node_feat - prev_feat)**2).sum()) / self.sigma**2)
            sim = np.inner(node_feat, prev_feat)/(np.linalg.norm(node_feat)*np.linalg.norm(prev_feat))
            p_include = 1 - sim**2
            if p_include < min_prob:
                min_prob = p_include
            if p_include > max_prob:
                max_prob = p_include
            if rng.uniform() <= p_include:
                #print "add one sample with probability: %.4f" % (p_include)
                sample.append(i)
        print "seqDPP: %d samples from [%d] ground set prob:[%.4f-%.4f]" % (len(sample), len(feats), min_prob, max_prob)    
        return sample
        
    def sample_seq_hierarchy(self, feats, rng=np.random):
        """
        Sequentially sample a list of items considering both diversity and quality. 
        The algorithm used is adapted from (Diverse Sequential Subset Selection
        for Supervised Video Summarization, NIPS 2014).
        """
        samples = []
        groundset = range(len(feats))
        min_prob_list = []
        max_prob_list = []
        while len(groundset) > 2:
            min_prob = 1.0
            max_prob = 0.0
            sample = []
            if rng.uniform() <= 0.5:
                sample.append(groundset[0])
            else:
                sample.append(groundset[1])
            for i in range(2, len(groundset)):
                node_feat = feats[groundset[i], :]
                prev_feat = feats[sample[-1],:]
                #sim = np.exp(-0.5 * (((node_feat - prev_feat)**2).sum()) / self.sigma**2)
                sim = np.inner(node_feat, prev_feat)/(np.linalg.norm(node_feat)*np.linalg.norm(prev_feat))
                p_include = 1 - sim**2
                if p_include < min_prob:
                    min_prob = p_include
                if p_include > max_prob:
                    max_prob = p_include
                if rng.uniform() <= p_include:
                    #print "add one sample with probability: %.4f" % (p_include)
                    sample.append(groundset[i])
            samples.append(sample)
            groundset = sample
            min_prob_list.append(min_prob)
            max_prob_list.append(max_prob)
        print "seqDPP (level-0): %d samples selected from [%d] ground set, include prob:[%.4f-%.4f]" % (len(samples[0]), len(feats), min_prob_list[0],max_prob_list[0])
        for i in range(1, len(samples)):        
            print "seqDPP (level-%d): %d samples selected from [%d] ground set, include prob:[%.4f-%.4f]" % (i, len(samples[i]), len(samples[i-1]), min_prob_list[i],max_prob_list[i])
            
        # get relations between adjacent levels
        edges = []
        for i in range(1, len(samples)):
            parent = samples[-i]
            child = samples[-i-1]
            L_sub = np.zeros((len(parent),len(child)), np.float32)
            # constrain the clustering to be temporally local
            for m in range(len(parent)):
                for n in range(len(child)):
                    #L_sub[m,n] = np.exp(-0.5 * (((feats[parent[m],:] - feats[child[n],:])**2).sum()) / self.sigma**2)
                    L_sub[m,n] = np.inner(feats[parent[m],:], feats[child[n],:])/(np.linalg.norm(feats[parent[m],:])*np.linalg.norm(feats[child[n],:]))
                    # constrain the clustering to be temporally local
                    # lower-bound child
                    if m-1 >= 0 and child[n] <= parent[m-1]:
                        L_sub[m,n] = 0
                    # upper-bound child
                    if m+1 < len(parent) and child[n] >= parent[m+1]:
                        L_sub[m,n] = 0
            idx = np.argmax(L_sub, axis=0)
            edge = {}
            for j in range(len(idx)):
                if edge.has_key(parent[idx[j]]):
                    edge[parent[idx[j]]].append(child[j])
                else:
                    edge[parent[idx[j]]] = [child[j]]
            edges.append(edge)                
        
        return edges
        
    def sample_seq_hierarchy_stream(self, feats, rng=np.random):
        """
        Sequentially sample a list of items considering both diversity and quality. 
        The algorithm used is adapted from (Diverse Sequential Subset Selection
        for Supervised Video Summarization, NIPS 2014).
        """
        Level = 1
        samples = [[0]]
        probs = [[1.0]]
        for i in range(1, len(feats)):
            node_feat = feats[i, :]
            l = 0
            while (l<Level):
                prev_feat = feats[samples[l][-1],:]
                #sim = np.exp(-0.5 * (((node_feat - prev_feat)**2).sum()) / self.sigma**2)
                sim = np.inner(node_feat, prev_feat)/(np.linalg.norm(node_feat)*np.linalg.norm(prev_feat))
                p_include = 1 - sim**2
                probs[l].append(p_include)
                if rng.uniform() <= p_include:
                    #print "level %d: add one sample with probability: %.4f" % (l, p_include)
                    samples[l].append(i)
                    l += 1
                else:
                    break
            if l == Level:
                Level += 1
                samples.append([i])
                probs.append([1.0])
        for i in range(Level):  
            if len(samples[i])<2:
                break
            print "seqDPP-stream (level-%d): %d samples from [%d] ground set, prob:[%.4f-%.4f]"\
                % (i, len(samples[i]), len(feats), min(probs[i][1:]), max(probs[i][1:]))    
        #draw_prob([range(len(feats))]+samples, probs, 1)
        #return

        # get relations between adjacent levels
        edges = []
        for i in range(1, len(samples)):
            parent = samples[-i]
            child = samples[-i-1]
            L_sub = np.zeros((len(parent),len(child)), np.float32)
            # constrain the clustering to be temporally local
            for m in range(len(parent)):
                for n in range(len(child)):
                    #L_sub[m,n] = np.exp(-0.5 * (((feats[parent[m],:] - feats[child[n],:])**2).sum()) / self.sigma**2)
                    L_sub[m,n] = np.inner(feats[parent[m],:], feats[child[n],:])/(np.linalg.norm(feats[parent[m],:])*np.linalg.norm(feats[child[n],:]))
                    # constrain the clustering to be temporally local
                    # lower-bound child
#                    if m-1 >= 0 and child[n] <= parent[m-1]:
#                        L_sub[m,n] = 0
                    # upper-bound child
#                    if m+1 < len(parent) and child[n] >= parent[m+1]:
#                        L_sub[m,n] = 0
            idx = np.argmax(L_sub, axis=0)
            edge = {}
            for j in range(len(idx)):
                if edge.has_key(parent[idx[j]]):
                    edge[parent[idx[j]]].append(child[j])
                else:
                    edge[parent[idx[j]]] = [child[j]]
            edges.append(edge)                
        
        return edges
    
    def sample_local_global(self, feats, rng=np.random):
        """
        seqDPP followed by standard stream DPP
        """
        rng.seed(0)
        # local seqDPP
        Level = 1
        samples = [[0]]
        probs = [[1.0]]
        for i in range(1, len(feats)):
            node_feat = feats[i, :]
            l = 0
            while (l<Level):
                prev_feat = feats[samples[l][-1],:]
                #sim = np.exp(-0.5 * (((node_feat - prev_feat)**2).sum()) / self.sigma**2)
                sim = np.inner(node_feat, prev_feat)/(np.linalg.norm(node_feat)*np.linalg.norm(prev_feat))
                p_include = 1 - sim**2
                probs[l].append(p_include)
                if rng.uniform() <= p_include:
                    #print "level %d: add one sample with probability: %.4f" % (l, p_include)
                    samples[l].append(i)
                    l += 1
                else:
                    break
            if l == Level:
                Level += 1
                samples.append([i])
                probs.append([1.0])
        compact_level = 0
        for i in range(Level):  
            if min(probs[i][1:])>0.5:
                compact_level = i-1
                break
            print "seqDPP (level-%d): %d samples from [%d] ground set, prob:[%.4f-%.4f]"\
                % (i, len(samples[i]), len(feats), min(probs[i][1:]), max(probs[i][1:]))
        #return samples[compact_level] # only for get_compact_set()
        
        # global stream DPP
        compact_sample = samples[compact_level]
        new_feats = feats[compact_sample, :]
        Level = 1
        new_samples = [[0]]
        new_probs = [[1.0]]
        for i in range(1, len(new_feats)):
            l = 0
            while (l<Level):
                L = build_cosine_similary_matrix(new_feats[new_samples[l]+[i],:])
                L_Y = L[0:len(L)-1,:]
                L_Y = L_Y[:,0:len(L)-1]
                L_Y_inv = np.linalg.inv(L_Y)
                c_u = L[len(L)-1, len(L)-1]
                b_u = L[0:len(L)-1, :]
                b_u = b_u[:, -1]
        
                p_include = min(1, (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
                new_probs[l].append(p_include)
                if rng.uniform() <= p_include:
                    #print "level %d: add one sample with probability: %.4f" % (l, p_include)
                    new_samples[l].append(i)
                    l += 1
                else:
                    break
            if l == Level:
                Level += 1
                new_samples.append([i])
                new_probs.append([1.0])
        diverse_level = 0
        for i in range(Level):            
            if len(new_samples[i])<2:
                break
            if min(new_probs[i][1:])>0.33:
                diverse_level = i-1
                break
            print "streamDPP (level-%d): %d samples from [%d] ground set, prob:[%.4f-%.4f]"\
                % (i+compact_level+1, len(new_samples[i]), len(feats), min(new_probs[i][1:]), max(new_probs[i][1:]))
        
        samples = samples[0:compact_level+1]
        for l in range(Level):
            sample = [compact_sample[v] for v in new_samples[l]]
            samples.append(sample)
        return samples[compact_level+diverse_level+1]
            
        # get relations between adjacent levels
        edges = []
        for i in range(1, len(samples)):
            parent = samples[-i]
            child = samples[-i-1]
            L_sub = np.zeros((len(parent),len(child)), np.float32)
            # constrain the clustering to be temporally local
            for m in range(len(parent)):
                for n in range(len(child)):
                    L_sub[m,n] = np.inner(feats[parent[m],:], feats[child[n],:])/(np.linalg.norm(feats[parent[m],:])*np.linalg.norm(feats[child[n],:]))
                    # constrain the clustering to be temporally local
                    # lower-bound child
#                    if m-1 >= 0 and child[n] <= parent[m-1]:
#                        L_sub[m,n] = 0
                    # upper-bound child
#                    if m+1 < len(parent) and child[n] >= parent[m+1]:
#                        L_sub[m,n] = 0
            idx = np.argmax(L_sub, axis=0)
            edge = {}
            for j in range(len(idx)):
                if edge.has_key(parent[idx[j]]):
                    edge[parent[idx[j]]].append(child[j])
                else:
                    edge[parent[idx[j]]] = [child[j]]
            edges.append(edge)                
        
        return edges
        
    def sample_k_nips2013(self, L, K, max_nb_iterations=1000, rng=np.random):
        """
        Sample a list of k items from a DPP defined
        by the similarity matrix L. The algorithm
        is iterative and runs for max_nb_iterations.
        The algorithm used is from
        (Fast Determinantal Point Process Sampling with
        Application to Clustering, Byungkon Kang, NIPS 2013)
        """
        K = int(K*len(L)/776)
        initial = rng.choice(range(len(L)), size=K, replace=False)
        X = [False] * len(L)
        for i in initial:
            X[i] = True
        X = np.array(X)
    
        min_prob = 1.0
        max_prob = 0.0
        for i in range(max_nb_iterations):
            u = rng.choice(np.arange(len(L))[X])
            v = rng.choice(np.arange(len(L))[~X])
            Y = X.copy()
            Y[u] = False
            L_Y = L[Y, :]
            L_Y = L_Y[:, Y]
            L_Y_inv = np.linalg.inv(L_Y)
    
            c_v = L[v:v+1, :]
            c_v = c_v[:, v:v+1]
            b_v = L[Y, :]
            b_v = b_v[:, v:v+1]
            c_u = L[u:u+1, :]
            c_u = c_u[:, u:u+1]
            b_u = L[Y, :]
            b_u = b_u[:, u:u+1]
    
            p = min(1, 0.5*(c_v - np.dot(np.dot(b_v.T, L_Y_inv), b_v))/
                       (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))    
            if p < min_prob:
                min_prob = p
            if p > max_prob:
                max_prob = p
            if rng.uniform() <= p:
                #print "swap with probability: %.4f (prob rate: %.4f)" % (p, (c_v - np.dot(np.dot(b_v.T, L_Y_inv), b_v))/(c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
                X = Y[:]
                X[v] = True
        #return np.array(items)[X]
        sample = []
        for i in range(len(L)):
            if X[i] == True:
                sample.append(i)
        
        print "k-DPP: %d samples from [%d] ground set [sigma:%.2f,K:%d] prob:[%.4f-%.4f]" % (len(sample), len(L), self.sigma, K, min_prob, max_prob)
        return sample
        
    def sample_k_icml2016(self, L, K, max_nb_iterations=1000, rng=np.random):
        """
        Sample a list of k items from a DPP defined
        by the similarity matrix L. The algorithm
        is iterative and runs for max_nb_iterations.
        The algorithm used is from
        (Fast DPP Sampling for Nystrom with Applications
        to Kernel Methods, ICML 2016)
        !!!determinant is too small (almost zero when K=200)!!!
        """
        initial = rng.choice(range(len(L)), size=K, replace=False)
        X = [False] * len(L)
        for i in initial:
            X[i] = True
        X = np.array(X)
        L_X = L[X, :]
        L_X = L_X[:, X]
        L_X_det = np.linalg.det(L_X)
    
        for i in range(max_nb_iterations):
            u = rng.choice(np.arange(len(L))[X])
            v = rng.choice(np.arange(len(L))[~X])
            Y = X.copy()
            Y[u] = False
            Y[v] = True
            L_Y = L[Y, :]
            L_Y = L_Y[:, Y]
            L_Y_det = np.linalg.det(L_Y)
    
            p = min(1, L_Y_det/(L_Y_det+L_X_det+0.0001)) 
            print "swap with probability: %.4f (prob rate: %.4f)" % (p, L_Y_det/(L_Y_det+L_X_det+0.0001))
            if rng.uniform() <= p:
                #print "swap with probability: %.4f (prob rate: %.4f)" % (p, L_Y_det/(L_Y_det+L_X_det+0.0001))
                X = Y[:]
                L_X_det = L_Y_det

        sample = []
        for i in range(len(L)):
            if X[i] == True:
                sample.append(i)
        print "%d samples selected from [%d] ground set [sigma:%.2f,K:%d]" % (len(sample), len(L), self.sigma, K)
        return sample

    def run_multiple(self, feats):
        """
        sample diverse sets at multiple levels
        feats: feature array
        sigma: Gaussian standard covariance
        K: cluster number at base level
        """
        start_time = time.time()
        L = build_similary_matrix(exp_quadratic(sigma=self.sigma), feats)
        end_time = time.time()
        print "runing time for calculating similarity matrix: %s s" % (end_time-start_time)
        print "sampled similarity: Avg:%.4f,L[0,30]:%.4f,L[0,100]:%.4f,L[0,300]:%.4f" % ((np.sum(L)-len(L))/(len(L)**2-len(L)), L[0,30], L[0,100], L[0,300])
        
        # sampling at different levels
        k = int(self.K*len(feats)/776)
        cluster_idxes = []
        L_sub = L
        prev_idx = range(len(L))
        while k >5:
            # local index from previous level samples
            idx = self.sample_k_nips2013(L_sub, k, max_nb_iterations=len(L_sub))
            # transfer to global index of original samples
            cluster_idx = [prev_idx[v] for v in idx]
            cluster_idxes.append(cluster_idx)
            k = int(k*0.7)
            prev_idx = cluster_idx
            L_sub = L_sub[idx, :]
            L_sub = L_sub[:, idx]
        #return
        # get relations between adjacent levels
        edges = []
        for i in range(1, len(cluster_idxes)):
            parent = cluster_idxes[-i]
            child = cluster_idxes[-i-1]
            L_sub = L[parent,:]
            L_sub = L_sub[:,child]
            # constrain the clustering to be temporally local
#            for m in range(len(parent)):
#                for n in range(len(child)):
                    # lower-bound child
#                    if m-1 >= 0 and child[n] <= parent[m-1]:
#                        L_sub[m,n] = 0
                    # upper-bound child
#                    if m+1 < len(parent) and child[n] >= parent[m+1]:
#                        L_sub[m,n] = 0
            idx = np.argmax(L_sub, axis=0)
            edge = {}
            for j in range(len(idx)):
                if edge.has_key(parent[idx[j]]):
                    edge[parent[idx[j]]].append(child[j])
                else:
                    edge[parent[idx[j]]] = [child[j]]
            edges.append(edge)                
        
        return edges

    def run(self, feats):
        """
        sample a diverse subset 
        feats: feature array
        sigma: Gaussian standard covariance
        """
        start_time = time.time()
        L = build_similary_matrix(exp_quadratic(sigma=self.sigma), feats)        
        end_time = time.time()
        print "runing time for calculating similarity matrix: %s s" % (end_time-start_time)
        print "sampled similarity: Avg:%.4f,L[0,30]:%.4f,L[0,100]:%.4f,L[0,300]:%.4f" % ((np.sum(L)-len(L))/(len(L)**2-len(L)), L[0,30], L[0,100], L[0,300])
        sampled = self.sample_k_nips2013(L, self.K, max_nb_iterations=len(feats))
        #sampled = self.sample_k_icml2016(L, self.K, max_nb_iterations=len(feats))
        #sampled = self.sample_fast(feats, L, max_nb_iterations=len(feats))
        #sampled = self.sample_standard(feats, L, max_nb_iterations=len(feats))
        return sampled

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #test()
    #sys.exit(0)
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    z = np.array(list(product(x, y)))
    sigmas = [0.0001, 0.1, 1, 2, 10]
    dpp = DPP()
    k = 1
    for sigma in sigmas:
        plt.subplot(1, len(sigmas) + 1, k)
        L = build_similary_matrix(exp_quadratic(sigma=sigma), z)
        selected_by_dpp = dpp.sample(z, L)
        plt.scatter(selected_by_dpp[:, 0], selected_by_dpp[:, 1])
        plt.title("DPP(sigma={0})".format(sigma))
        k += 1

    plt.subplot(1, len(sigmas) + 1, k)
    selected_by_random = z[np.random.choice((True, False),
                           size=len(z))]
    plt.scatter(selected_by_random[:, 0], selected_by_random[:, 1])
    plt.title("random")
    plt.show()