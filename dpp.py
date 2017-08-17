# dpp.py
import numpy as np
# import scipy.io as sio
# d = sio.loadmat('d.mat')['d']
# da = d.tolist()
# feats = da[0]

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


def sample_hierarchy_stream(feats, feature_len, seq_len, rng=np.random):
    """
    Sample a hierarchical list of items from a DPP defined by the similarity matrix L.
    The algorithm is adapted to streaming setting with only insertion.
    """
    a = []
    feature_len = int(feature_len);
    for i in range(int(len(feats)/feature_len)):
        a.append(feats[i*feature_len:(i+1)*feature_len])
    feats = np.matrix(a)
    Level = 1
    samples = [[0]]
    probs = [[1.0]]
    pro = []
    for i in range(1, len(feats)):
        l = 0
        while (l < Level):
            if len(pro)<=l:
                pro.append({})
            L = build_cosine_similary_matrix(feats[samples[l] + [i], :])
            L_Y = L[0:len(L) - 1, :]
            L_Y = L_Y[:, 0:len(L) - 1]
            L_Y_inv = np.linalg.inv(L_Y)
            c_u = L[len(L) - 1, len(L) - 1]
            b_u = L[0:len(L) - 1, :]
            b_u = b_u[:, -1]

            p_include = min(1, (c_u - np.dot(np.dot(b_u.T, L_Y_inv), b_u)))
            probs[l].append(p_include)
            if rng.uniform() <= p_include:
                #print "level %d: add one sample with probability: %.4f" % (l, p_include)
                pro[l][i] = p_include
                samples[l].append(i)
                l += 1
            else:
                break
        if l == Level: #and len(samples[l-1])>=seq_len:
            Level += 1
            samples.append([i])
            probs.append([1.0])

    for i in range(Level):
        if len(samples[i+1]) < seq_len <= len(samples[i]) :
            a1 = sorted(pro[i].items(), key=lambda item: item[1])
            a1 = a1[-15:]
            sa = sorted([i[0] for i in a1])
        #if len(samples[i])<2:
            break
        print "streamDPP (level-%d): %d samples from [%d] ground set, prob:[%.4f-%.4f]" \
              % (i, len(samples[i]), len(feats), min(probs[i][1:]), max(probs[i][1:]))
    return sa