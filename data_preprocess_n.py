from __future__ import division
import sys
import re
import math
import numpy as np
import os
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from pyrouge import Rouge155
from sklearn.decomposition import TruncatedSVD
from nltk import word_tokenize
import nltk 
import codecs
import ConfigParser
import pickle
from nltk.stem.porter import PorterStemmer
import copy
import random

global_conf = ConfigParser.ConfigParser()
idf_dict = {}
df_dict = {}

class TopicInfo:
  def __init__(self):
    self.topic_sen = ''
    self.topic_words = []
    self.topic_bwords = set([])
    self.global_task_id = ''
    self.topic_sim = []
    self.word_used = []
    self.stemwords = []
    self.smatrix = None
    self.concepts = {}

tpinfo = TopicInfo()

def StatConcepts(senlist):
  tpinfo.concepts.clear()
  cset = set()
  for sen in senlist:
    if sen.sen_order.find('h') >= 0:
      for w in cset:
        if w not in tpinfo.concepts:
          tpinfo.concepts[w] = 1
        else:
          tpinfo.concepts[w] += 1
      cset.clear()
    if sen.topic_diversity > 0.0:
      cset = cset | sen.sen_bwords
  for w in cset:
    if w not in tpinfo.concepts:
      tpinfo.concepts[w] = 1
    else:
      tpinfo.concepts[w] += 1
  
  #del the bigram less than 3 times
  tmp_concepts = {}
  for w in tpinfo.concepts:
    if tpinfo.concepts[w] >= 3:
      tmp_concepts[w] = tpinfo.concepts[w]
  tpinfo.concepts = tmp_concepts

  citems = tpinfo.concepts.items()
  clen = len(citems)
  print '[concepts]:',clen
  sortlist = sorted(citems, key=lambda x:x[1], reverse=True)
  for sl in sortlist:
    print '%s:%d '%(sl[0], sl[1]),
  print ''
  """
  slen = len(senlist)
  for i in range(0, slen):
    c_scores = 0
    for w in senlist[i].sen_bwords:
      if w in tpinfo.concepts:
        c_scores += tpinfo.concepts[w]
    senlist[i].concepts_found = c_scores
    #print i,c_scores,senlist[i].sen_order
  """
  return

def CalConceptScore(sen):
  c_scores = 0
  for w in sen.sen_bwords:
    if w in tpinfo.concepts:
      c_scores += tpinfo.concepts[w]
  return c_scores

def CalAddConceptScore(sen, cused):
  c_scores = 0
  for w in sen.sen_bwords:
    if w not in cused and w in tpinfo.concepts:
      c_scores += tpinfo.concepts[w]
      cused.add(w)
  return c_scores


def UpdateConceptWeight(sen):
  for w in sen.sen_bwords:
    if w in tpinfo.concepts:
      tpinfo.concepts[w] *= 0.7
  return

class sen_elems:
  def __init__(self):
    self.sen_id = 0
    self.sen_str = ''
    self.sen_len = 0
    self.tol_words = 0
    self.uniq_word_num = 0
    self.sen_words = []
    self.stemwords = []
    self.word_dict = {}
    self.sen_vector = {}
    self.con_vector = []
    self.sen_label = -1
    self.word_found = 0
    self.word_used = []
    self.word_notused = []
    self.sen_bwords = []
    self.topic_diversity = 0
    self.weight = 1
    self.sen_order = ''
    self.rank_weight = 0
    self.concepts_found = 0

stoplist = set()
#stoplist = set(nltk.corpus.stopwords.words('english'))
def ReadStoplist():
  infile = codecs.open('stopwords.english', 'r', 'utf-8')
  for line in infile:
    sword = line.strip()
    stoplist.add(sword)
  infile.close()
  print '[Stopwords]:', len(stoplist)
  return

def CutWords(sen_str, stem=False):
  words = word_tokenize(sen_str.lower())
  #wlen = len(words)
  wlen = len(sen_str.lower().split())
  final_words = []
  for word in words:
    if word != None and word != '' and word not in [',','.','!','\'', '"']:
      if word in stoplist:
        continue
      final_words.append(word)
  return final_words, wlen

st = PorterStemmer()
def StemWords(words):
  swords = [st.stem(w) for w in words]
  return swords

def Bigram_words(words):
  wset = set([])
  wlen = len(words)
  #for w in words:
  #  wset.add(w)
  for i in range(0, wlen - 1):
    wset.add(words[i] + '@' + words[i+1])
  return wset

spattern = re.compile('[.,]')
def SplitSen(sen_str):
  sen_str = spattern.sub('@@', sen_str)
  sens = sen_str.split('@@')
  return sens

def ReadDucText(in_path):
  # read sentences
  file_in = codecs.open(in_path, 'r', 'utf-8')
  fkey = in_path.split('/')[-1].split('-')[0]
  senlist = []
  cnt = 0
  tmp_date = ''
  weight_sum = 0
  for line in file_in:
    asen = line.lower().strip()
    sens = [asen] #SplitSen(asen)
    for tsen in sens:
      svals = tsen.split('#@#')
      p_or_h = svals[0]
      ph_cnt = '-1'
      if p_or_h == 'h':
        tmp_date = svals[1]
        ph_cnt = '1'
      elif p_or_h == 'p':
        ph_cnt = svals[1]
      st_cnt = svals[2]
      sen = svals[3]
      if sen != None and sen != '':
        senelem = sen_elems()
        senelem.sen_str = sen
        senelem.sen_words, senelem.tol_words = CutWords(sen)
        if senelem.tol_words < 4:
          continue
        senelem.sen_id = cnt
        cnt += 1
        senelem.sen_len = len(senelem.sen_words)
        senelem.stemwords = StemWords(senelem.sen_words)
        senelem.sen_bwords = Bigram_words(senelem.stemwords)
        senelem.sen_order = tmp_date + p_or_h + ph_cnt.zfill(3) + st_cnt.zfill(3)
        senelem.weight = 10 if p_or_h == 'h' else 1
        senelem.weight = senelem.weight / math.log(int(st_cnt)+1,2)
        weight_sum += senelem.weight
        senlist.append(senelem)
  file_in.close()
  slen = len(senlist)
  for i in range(0, slen):
    senlist[i].weight *= (slen/weight_sum)

  has_topic = global_conf.has_option('topic', 'topic_on')
  if has_topic:
    topic_file = global_conf.get('topic', 'topic_file')
    tfile = codecs.open(topic_file, 'r', 'utf-8')
    for line in tfile:
      vals = line.strip().split('####')
      if vals[0] == fkey:
        tpinfo.topic_sen = vals[1] + ' ' + vals[2]
        tpinfo.topic_words, tmp_len = CutWords(tpinfo.topic_sen)
        tpinfo.stemwords = StemWords(tpinfo.topic_words)
        tpinfo.topic_bwords = Bigram_words(tpinfo.stemwords)
        print "[Topic]:",tpinfo.topic_sen
        print "[Words]:",tpinfo.stemwords
        break
    tfile.close()
    slen = len(senlist)
    max_d = -1
    for i in range(0, slen):
      senlist[i].topic_diversity = len(set(tpinfo.topic_words) & set(senlist[i].sen_words))
      max_d = max(senlist[i].topic_diversity, max_d)
      #print senlist[i].topic_diversity
    filter_senlist = []
    for i in range(0, slen):
      senlist[i].topic_diversity /= max_d
      if senlist[i].topic_diversity > 0.0:
        filter_senlist.append(senlist[i])
    #for i in range(0, len(filter_senlist)):
    #  filter_senlist[i].sen_id = i
    #senlist = filter_senlist
    StatConcepts(senlist)
  return senlist

def ReadText(in_path):
  # read sentences
  file_in = codecs.open(in_path, 'r', 'utf-8')
  fkey = in_path.split('/')[-1].split('-')[0]
  senlist = []
  cnt = 0
  for line in file_in:
    asen = line.lower().strip()
    sens = [asen] #SplitSen(asen)
    for sen in sens:
      if sen != None and sen != '':
        senelem = sen_elems()
        senelem.sen_str = sen
        senelem.sen_words, senelem.tol_words = CutWords(sen)
        if senelem.tol_words < 4:
          continue
        senelem.sen_id = cnt
        cnt += 1
        senelem.sen_len = len(senelem.sen_words)
        senelem.stemwords = StemWords(senelem.sen_words)
        senelem.sen_bwords = Bigram_words(senelem.stemwords)
        senlist.append(senelem)
  file_in.close()
  has_topic = global_conf.has_option('topic', 'topic_on')
  if has_topic:
    topic_file = global_conf.get('topic', 'topic_file')
    tfile = codecs.open(topic_file, 'r', 'utf-8')
    for line in tfile:
      vals = line.strip().split('####')
      if vals[0] == fkey:
        tpinfo.topic_sen = vals[1] + ' ' + vals[2]
        tpinfo.topic_words, tmp_len = CutWords(tpinfo.topic_sen)
        tpinfo.stemwords = StemWords(tpinfo.topic_words)
        tpinfo.topic_bwords = Bigram_words(tpinfo.stemwords)
        print "[Topic]:",tpinfo.topic_sen
        print "[Words]:",tpinfo.stemwords
        break
    tfile.close()
    
    slen = len(senlist)
    max_d = -1
    for i in range(0, slen):
      #senlist[i].topic_diversity = len(tpinfo.topic_bwords & senlist[i].sen_bwords)
      senlist[i].topic_diversity = len(set(tpinfo.topic_words) & set(senlist[i].sen_words))
      max_d = max(senlist[i].topic_diversity, max_d)
      #print senlist[i].topic_diversity
    filter_senlist = []
    for i in range(0, slen):
      senlist[i].topic_diversity /= max_d
      if senlist[i].topic_diversity > 0.0:
        filter_senlist.append(senlist[i])
    #for i in range(0, len(filter_senlist)):
    #  filter_senlist[i].sen_id = i
    #senlist = filter_senlist
  return senlist

def PrintSenlist(senlist):
  for sen in senlist:
    #print '['+str(sen.sen_id)+']',sen.sen_str,sen.sen_len,sen.sen_label
    print sen.sen_label
    #for word in sen.word_dict:
    #  print word+' '+str(sen.word_dict[word]),
    #print

def TfIdfVector(osen):
  #TF
  slen = len(osen)
  for i in range(0, slen):
    wcnt = 0
    for word in osen[i].sen_words:
      if word not in osen[i].word_dict:
        osen[i].word_dict[word] = 1
      else:
        osen[i].word_dict[word] += 1
      wcnt += 1
    osen[i].uniq_word_num = wcnt #len(osen[i].word_dict)
    
    #for word in osen[i].word_dict:
    #  osen[i].word_dict[word] /= osen[i].sen_len
 
  #IDF
  #idf_dict = {}
  idf_dict.clear()
  df_dict.clear()
  idf_cnt = 0
  for i in range(0, slen):
    for word in osen[i].word_dict:
      if word not in idf_dict:
        idf_dict[word] = [idf_cnt,1]
        df_dict[word] = 1
        idf_cnt += 1
      else:
        idf_dict[word][1] += 1
        df_dict[word] += 1
  
  ofile = codecs.open('idf.txt','w','utf-8')
  for word in idf_dict:
    ostr = str(idf_dict[word][0])+' '+word+' '+str(idf_dict[word][1]) +'\n'
    idf_dict[word][1] = math.log(slen/(idf_dict[word][1] + 1) )
    ofile.write(ostr)
  ofile.close()
  
  print '[IDF WORD SIZE]: %d'%(len(idf_dict))
  #TF*IDF
  for i in range(0, slen):
    for word in osen[i].word_dict:
      osen[i].sen_vector[idf_dict[word][0]] = idf_dict[word][1] * osen[i].word_dict[word]
  return

def TranSenToSparseMatrix(senlist, f_swap = False):
  row = []
  col = []
  val = []
  slen = len(senlist)
  for i in range(0, slen):
    for wid in senlist[i].sen_vector:
      if f_swap:
        row.append(wid)
        col.append(i)
      else:
        row.append(i)
        col.append(wid)
      val.append(senlist[i].sen_vector[wid])
      #print i,id_val[0],id_val[1]
  
  row = np.array(row)
  col = np.array(col)
  val = np.array(val)

  sparse_matrix = coo_matrix((val, (row, col)))
  return sparse_matrix

def LSA_Vector(senlist, dim=100):
  if len(senlist) < dim:
    dim = int(0.8 * len(senlist))
  sparse_matrix = TranSenToSparseMatrix(senlist)
  svd = TruncatedSVD(n_components=dim, random_state=1)
  omatrix = svd.fit_transform(sparse_matrix)
  #print(svd.explained_variance_ratio_)
  #print(svd.explained_variance_ratio_.sum())
  #print(svd.components_.shape)
  print len(senlist),omatrix.shape
  slen = len(senlist)
  for i in range(0, slen):
    senlist[i].con_vector = omatrix[i,:]
  return

def NormalizeVector(vec):
  vlen = len(vec)
  vsum = 0
  for i in range(0, vlen):
    vsum +=  vec[i] * vec[i]
  vsum = math.sqrt(vsum)
  for i in range(0, vlen):
    vec[i] = vec[i] / vsum
  return vec

def LoadWordVector(filepath):
  infile = open(filepath, 'r')
  w2vec = {}
  dim = -1
  for line in infile:
    vals = line.strip().split()
    w2vec[vals[0].lower()] = map(float,vals[1:])
  infile.close()
  for word in w2vec:
    dim = len(w2vec[word])
    break
  print "load word2vec vectors:",len(w2vec),dim
  #print w2vec['what']
  return [w2vec,dim]

def AddVector(avec, bvec, weight=1):
  alen = len(avec)
  for i in range(0, alen):
    avec[i] += bvec[i] * weight
  return 

def Word2Vec_Vector(senlist, w2vec, dim):
  slen = len(senlist)
  cnt = 0
  not_found = 0
  for i in range(0, slen):
    senlist[i].con_vector = [ 0.0 for j in range(0,dim)]
    #tags = dict(nltk.pos_tag([ w for w in senlist[i].word_dict]) )
    for word in senlist[i].word_dict:
      if len(word)>2 and word.isalpha and word not in stoplist:
        if word in w2vec:
          #if tags[word][0] not in ['N','V', 'J']:
          #  continue
          AddVector(senlist[i].con_vector, w2vec[word])    
          #AddVector(senlist[i].con_vector, w2vec[word], senlist[i].sen_vector[idf_dict[word][0]])    
          cnt += 1
          senlist[i].word_found += 1
          senlist[i].word_used.append(word)
        else:
          senlist[i].word_notused.append(word)
          not_found += 1
    """
    ssum = 0
    for j in range(0, dim):
      ssum += senlist[i].con_vector[j] * senlist[i].con_vector[j] 
    ssum = math.sqrt(ssum)
    if ssum == 0:
      continue
    for j in range(0, dim):
      senlist[i].con_vector[j] /= ssum
    """
    #print word
    #print senlist[i].con_vector
  print '[not found word cnt in w2v]:',not_found
  return

def Word2VecReduction(senlist, w2vec, ratio):
  slen = len(senlist)
  word_matrix = []
  word2label = {}
  idx2word = {}
  useword = set([])
  cnt = 0
  for i in range(0, slen):
    for word in senlist[i].word_used:
      if word not in useword: #and word in w2vec:
        idx2word[cnt] = word
        cnt += 1
        useword.add(word)
        word_matrix.append(w2vec[word])
  wlen = len(useword)
  print "use words:", wlen
  
  nclusters = max(int(0.9*wlen), 100)
  print nclusters
  AgloCluster = AgglomerativeClustering(n_clusters=nclusters,linkage="average", affinity='cosine')
  AgloCluster.fit(word_matrix)
  AgloCluster_labels = AgloCluster.labels_
  
  for i in range(0, wlen):
    word2label[idx2word[i]] = AgloCluster_labels[i]

  for i in range(0, slen):
    senlist[i].sen_words = [ str(word2label[w]) for w in senlist[i].word_used]
    senlist[i].word_dict = {}
    #print senlist[i].sen_words
  return

def Load_RAE_vector(rae_file, senlist):
  infile = open(rae_file, 'r')
  i = 0
  for line in infile:
    senlist[i].con_vector = map(float, line.strip().split(','))
    i += 1
  infile.close()
  print i
  return

def TfIdfBaseSentenceClustering(senlist):
  slen = len(senlist)
  sparse_matrix = TranSenToSparseMatrix(senlist)
  #print sparse_matrix.todense()
  k_means = KMeans(init='k-means++', n_clusters=int(0.2*slen), n_init=6, n_jobs=6, random_state=1)
  #k_means = KMeans(init='random', n_clusters=2, n_init=10)
  k_means.fit(sparse_matrix)
  k_means_labels = k_means.labels_
  
  for i in range(0, slen):
    senlist[i].sen_label = k_means_labels[i]
  return

def ConBaseSentenceClustering(senlist):
  slen = len(senlist)
  dense_matrix = []
  for i in range(0, slen):
    dense_matrix.append(senlist[i].con_vector)
  k_means = KMeans(init='k-means++', n_clusters=int(0.2*slen), n_init=10, n_jobs=6, random_state=1)
  #k_means = KMeans(init='random', n_clusters=2, n_init=10)
  k_means.fit(dense_matrix)
  k_means_labels = k_means.labels_
  
  for i in range(0, slen):
    senlist[i].sen_label = k_means_labels[i]
    #print senlist[i].sen_id,senlist[i].sen_label
  return

def SimSentenceClustering(pre_matrix, senlist):
  slen = len(senlist)
  nclusters = int(0.2 * slen)
  AgloCluster = AgglomerativeClustering(n_clusters=nclusters,linkage="average", affinity='precomputed')
  AgloCluster.fit(pre_matrix)
  AgloCluster_labels = AgloCluster.labels_
  
  for i in range(0, slen):
    senlist[i].sen_label = AgloCluster_labels[i]
    #print senlist[i].sen_id,senlist[i].sen_label
  return

def BagOfWordSim(asen, bsen):
  cos_sum = 0
  for wid in asen.sen_vector:
    if wid in bsen.sen_vector:
      cos_sum += asen.sen_vector[wid] * bsen.sen_vector[wid]

  asum = 0
  bsum = 0
  for wid in asen.sen_vector:
    asum += asen.sen_vector[wid] * asen.sen_vector[wid]
  
  for wid in bsen.sen_vector:
    bsum += bsen.sen_vector[wid] * bsen.sen_vector[wid]
  if asum == 0 or bsum == 0:
    return 0
  else: 
    return (cos_sum / math.sqrt(asum * bsum) + 1 ) / 2

def ConVectorSim(asen, bsen, dis='cos'):
  vlen = len(asen.con_vector)
  if dis == 'cos':
    cos_sum = 0
    asum = 0
    bsum = 0
    for i in range(0, vlen):
      cos_sum += asen.con_vector[i] * bsen.con_vector[i]
      asum += asen.con_vector[i] * asen.con_vector[i]
      bsum += bsen.con_vector[i] * bsen.con_vector[i]
    if asum == 0  or bsum == 0:
      return 0
    else:
      return (cos_sum / math.sqrt(asum * bsum) + 1 ) / 2
  elif dis == 'euc':
    euc_sum = 0
    for i in range(0, vlen):
      dif = asen.con_vector[i] - bsen.con_vector[i]
      euc_sum += dif * dif
    return math.sqrt(euc_sum)

cache_dis = {}

def LoadCachedis(ftype):
  cachedir = '../cache/'
  cachefile = cachedir + global_task_id + '-' + ftype +'.dat'
  if os.path.exists(cachefile):
    global cache_dis
    cache_dis = pickle.load(open(cachefile, 'rb'))
  return

def DumpCachedis(ftype):
  cachedir = '../cache/'
  cachefile = cachedir + global_task_id + '-' + ftype +'.dat'
  pickle.dump(cache_dis, open(cachefile, 'wb'))
  return

def WordSim(aword, bword, w2v_dict, dis='cos'):
  akey = aword + u'@@' + bword
  bkey = bword + u'@@' + aword
  if akey in cache_dis:
    return cache_dis[akey]
  elif bkey in cache_dis:
    return cache_dis[bkey]
  
  avector = w2v_dict[aword]
  bvector = w2v_dict[bword]
  vlen = len(avector)
  if dis == 'cos':
    cos_sum = 0
    asum = 0
    bsum = 0
    for i in range(0, vlen):
      cos_sum += avector[i] * bvector[i]
      asum += avector[i] * avector[i]
      bsum += bvector[i] * bvector[i]
    ans = 0
    if asum == 0  or bsum == 0:
      ans = 0
    else:
      ans = (cos_sum / math.sqrt(asum * bsum) + 1 ) / 2
    cache_dis[akey] = ans
    return ans
  elif dis == 'euc':
    euc_sum = 0
    for i in range(0, vlen):
      dif = avector[i] - bvector[i]
      euc_sum += dif * dif
    ans = math.sqrt(euc_sum)
    cache_dis[akey] = ans
    return ans


def N_WordSim(asen, bsen, w2v_dict, dtype='all', dis_method='cos'):
  sim = 0
  if dtype == 'match':
    aused = set([])
    bused = set([])
    wcnt = 0
    for aword in asen.word_used:
      tmp_closed = 0
      fword = '' 
      for bword in bsen.word_used:
        if bword not in bused:
          wsim = WordSim(aword, bword, w2v_dict, dis_method)
          if wsim > tmp_closed:
            tmp_closed = wsim 
            fword = bword
      if tmp_closed > 0 and fword != '':
        sim += tmp_closed #* math.sqrt(df_dict[aword] * df_dict[fword])
        #bused.add(fword)
        wcnt += 1

    for bword in bsen.word_used:
      tmp_closed = 0
      fword = '' 
      for aword in asen.word_used:
        if aword not in aused:
          wsim = WordSim(bword, aword, w2v_dict, dis_method)
          if wsim > tmp_closed:
            tmp_closed = wsim
            fword = aword
      if tmp_closed > 0 and fword != '':
        sim += tmp_closed #* math.sqrt(df_dict[aword] * df_dict[fword])
        #aused.add(fword)
        wcnt += 1
    if wcnt > 0:
      sim /= wcnt

  elif dtype == 'all':
    wcnt = 0
    for aword in asen.word_used:
      for bword in bsen.word_used:
        sim += WordSim(aword, bword, w2v_dict, dis_method)
        wcnt += 1
    if wcnt > 0:
      sim /= wcnt

  return sim

def GetNSimMatrix(sens, w2v_dict, dis='cos'):
  slen = len(sens)
  smatrix = [ [0 for i in range(0, slen)] for j in range(0, slen)]
  
  Nsim_type = global_conf.get('w2v', 'Nsim_type')
  Nsim_dis_method = 'cos'
  if global_conf.has_option('w2v', 'Nsim_dis_method'):
    Nsim_dis_method = global_conf.get('w2v', 'Nsim_dis_method')
  for i in range(0, slen):
    smatrix[i][i] = 1
    for j in range(0, i):
      smatrix[i][j] = N_WordSim(sens[i], sens[j],  w2v_dict, Nsim_type, Nsim_dis_method)
      smatrix[j][i] = smatrix[i][j]
  
  if dis == 'euc':
    max_dis = 0
    for i in range(0, slen):
      smatrix[i][i] = 0
      for j in range(0, i):
        max_dis = max(max_dis, smatrix[i][j])

    for i in range(0, slen):
      for j in range(0, slen):
        smatrix[i][j] = 1 - smatrix[i][j] / max_dis

  return smatrix

def GetNSimMatrixWithUnusedWord(sens, w2v_dict, dis='cos'):
  beta = 0.8
  slen = len(sens)
  smatrix = [ [0 for i in range(0, slen)] for j in range(0, slen)]

  Nsim_type = global_conf.get('w2v', 'Nsim_type')
  Nsim_dis_method = 'cos'
  if global_conf.has_option('w2v', 'Nsim_dis_method'):
    Nsim_dis_method = global_conf.get('w2v', 'Nsim_dis_method')
  ftype = Nsim_type + '-' + Nsim_dis_method
  
  cache_dis.clear()
  LoadCachedis(ftype)
  
  for i in range(0, slen):
    smatrix[i][i] = 1
    for j in range(0, i):
      smatrix[i][j] = N_WordSim(sens[i], sens[j],  w2v_dict, Nsim_type, Nsim_dis_method) #* beta + (1 - beta) * UnusedWordSim(sens[i], sens[j])
      smatrix[j][i] = smatrix[i][j]
  
  # copy the similarty matrix to the topic info
  tpinfo.smatrix = copy.deepcopy(smatrix)

  max_sim = 0
  min_sim = smatrix[0][0]
  for i in range(0, slen):
    smatrix[i][i] = 0
    for j in range(0, i):
      max_sim = max(max_sim, smatrix[i][j])
      min_sim = min(min_sim, smatrix[i][j])

  for i in range(0, slen):
    for j in range(0, i):
      smatrix[i][j] = (smatrix[i][j] - min_sim) / (max_sim - min_sim) * beta +(1 - beta) * UnusedWordSim(sens[i], sens[j])
      smatrix[j][i] = smatrix[i][j]


  if dis == 'euc':
    max_dis = 0
    for i in range(0, slen):
      smatrix[i][i] = 0
      for j in range(0, i):
        max_dis = max(max_dis, smatrix[i][j])

    for i in range(0, slen):
      for j in range(0, slen):
        smatrix[i][j] = 1 - smatrix[i][j] / max_dis
  
  DumpCachedis(ftype)
  
  return smatrix


def SentenceSimilarity(asen, bsen, dis='tfidf'):
  if dis == 'tfidf':
    return BagOfWordSim(asen, bsen)
  else:
    return ConVectorSim(asen, bsen, dis)

def GetSimMatrix(sens, dis='tfidf'):
  slen = len(sens)
  smatrix = [ [0 for i in range(0, slen)] for j in range(0, slen)]
  for i in range(0, slen):
    smatrix[i][i] = 1
    for j in range(0, i):
      smatrix[i][j] = SentenceSimilarity(sens[i], sens[j], dis)
      smatrix[j][i] = smatrix[i][j]
  
  if dis == 'euc':
    max_dis = 0
    for i in range(0, slen):
      smatrix[i][i] = 0
      for j in range(0, i):
        max_dis = max(max_dis, smatrix[i][j])

    for i in range(0, slen):
      for j in range(0, slen):
        smatrix[i][j] = 1 - smatrix[i][j] / max_dis
  #for i in range(0, slen):
  #  print ' '.join([str(round(x,2)) for x in smatrix[i]])
  return smatrix

def UnusedWordSim(asen, bsen):
  uw_sim = 0
  aset = set(asen.word_notused)
  bset = set(bsen.word_notused)
  if len(aset) > 0 or len(bset) > 0:
    uw_sim = len(aset & bset) / len(aset | bset)
  return uw_sim

def GetSimMatrixWithUnusedWord(sens, dis='tfidf'):
  beta = 0.8
  slen = len(sens)
  smatrix = [ [0 for i in range(0, slen)] for j in range(0, slen)]
  for i in range(0, slen):
    smatrix[i][i] = 1
    for j in range(0, i):
      smatrix[i][j] = SentenceSimilarity(sens[i], sens[j], dis) * beta + (1 - beta) * UnusedWordSim(sens[i], sens[j])
      smatrix[j][i] = smatrix[i][j]
  
  if dis == 'euc':
    max_dis = 0
    for i in range(0, slen):
      smatrix[i][i] = 0
      for j in range(0, i):
        max_dis = max(max_dis, smatrix[i][j])

    for i in range(0, slen):
      for j in range(0, slen):
        smatrix[i][j] = 1 - smatrix[i][j] / max_dis
  #for i in range(0, slen):
  #  print ' '.join([str(round(x,2)) for x in smatrix[i]])
  return smatrix

def CompCovery(sums, sens, smatrix):
  a_cov = 10 / len(sens)
  cov_sum = 0
  for sen in sens:
    part_sim = 0
    tol_sim = 0
    
    for s in sums:
      part_sim += smatrix[sen.sen_id][s.sen_id]
    
    for s in sens:
      tol_sim += smatrix[sen.sen_id][s.sen_id]

    cov_sum +=  min(part_sim, a_cov * tol_sim)
  return cov_sum

def CompDiversity(sums, sens, smatrix):
  div_sum = 0
  
  slen = len(sens)
  part = {}
  for i in range(0, slen):
    if sens[i].sen_label != -1:
      if sens[i].sen_label not in part:
        part[sens[i].sen_label] = set([i])
      else:
        part[sens[i].sen_label].add(i)
  
  sum_set = set([ sen.sen_id for sen in sums])

  for k in part:
    same_part = part[k] & sum_set
    d_sum = 0
    for j in same_part:
      for i in range(0, slen):
        d_sum += smatrix[i][j]
    div_sum += math.sqrt(d_sum  / slen)

  return div_sum 

def CompTopicDiversity(sums, sens, smatrix):
  div_sum = 0
  beta = 1.0
  slen = len(sens)
  part = {}
  for i in range(0, slen):
    if sens[i].sen_label != -1:
      if sens[i].sen_label not in part:
        part[sens[i].sen_label] = set([i])
      else:
        part[sens[i].sen_label].add(i)
  
  sum_set = set([ sen.sen_id for sen in sums])

  for k in part:
    same_part = part[k] & sum_set
    d_sum = 0
    for j in same_part:
      for i in range(0, slen):
        d_sum += smatrix[i][j]
      d_sum = d_sum / slen * beta + (1 - beta) * sens[j].topic_diversity
    div_sum += math.sqrt(d_sum)

  return div_sum 

def SubmodularFunction(sums, sens, smatrix, topic_on=False):
  if topic_on:
    return CompCovery(sums, sens, smatrix) , 10 * CompTopicDiversity(sums, sens, smatrix)
  else:
    return CompCovery(sums, sens, smatrix) , 10 * CompDiversity(sums, sens, smatrix)

def BagOfStemWordSim(asen, bsen):
  cos_sum = 0
  for wid in asen.sen_vector:
    if wid in bsen.sen_vector:
      cos_sum += asen.sen_vector[wid] * bsen.sen_vector[wid]

  asum = 0
  bsum = 0
  for wid in asen.sen_vector:
    asum += asen.sen_vector[wid] * asen.sen_vector[wid]
  
  for wid in bsen.sen_vector:
    bsum += bsen.sen_vector[wid] * bsen.sen_vector[wid]
  if asum == 0 or bsum == 0:
    return 0
  else: 
    return (cos_sum / math.sqrt(asum * bsum) + 1 ) / 2

def ManiFoldRanking(senlist, w2vec):
  tpinfo.word_used = []
  for w in tpinfo.topic_words:
    if w in w2vec:
      tpinfo.word_used.append(w)

  slen = len(senlist)
  max_sim = -1
  rmatrix = [ [0 for i in range(0, slen+1)] for j in range(0, slen+1)]
  for i in range(0, slen):
    rmatrix[i][slen] = N_WordSim(tpinfo, senlist[i], w2vec, 'match', 'cos')
    rmatrix[slen][i] = rmatrix[i][slen]
    max_sim = max(max_sim, rmatrix[i][slen])
    for j in range(0, i):
      rmatrix[i][j] = tpinfo.smatrix[i][j]
      rmatrix[j][i] = rmatrix[i][j]
      max_sim = max(max_sim, rmatrix[i][j])
  
  #normalize
  #"""
  max_sim = 0.65 * max_sim
  for i in range(0, slen+1):
    for j in range(0, i):
      if rmatrix[i][j] < max_sim:
        rmatrix[i][j] = rmatrix[j][i] = 0
  #"""
  for i in range(0, slen+1):
    rsum = sum(rmatrix[i])
    if rsum == 0.0:
      continue
    for j in range(0, slen+1):
      rmatrix[i][j] /= rsum
  #print rmatrix[slen]
  #page rank
  rmatrix = np.mat(rmatrix)
  y = [ 0 for i in range(0, slen) ] + [1]
  y = np.mat(y)
  y = y.T
  a_ = 0.6
  ft = np.mat([1 for i in range(0, slen+1)])
  ft = ft.T
  for itr in range(0, 250):
    nft =  rmatrix * ft * a_+ (1 - a_) * y
    ft = nft 
    #print 'itr',itr
  n_min = 10
  n_max = 0
  for i in range(0, slen):
    n_min = min(n_min, float(ft[i][0]))
    n_max = max(n_max, float(ft[i][0]))
    
  for i in range(0, slen):
    senlist[i].rank_weight = (float(ft[i][0]) - n_min) / (n_max - n_min)
    #print i,senlist[i].rank_weight
  return

def GreedySearch(sens, limit, smatrix):
  slen = len(sens)
  sums = []
  sel = set([])
  tmpinfo = {}
  max_sents = int(global_conf.get('greedysearch', 'max_sents'))
  max_words = 109 #int(global_conf.get('greedysearch', 'max_words'))
  topic_on = False
  if global_conf.has_option('topic', 'topic_on'):
    topic_on = global_conf.getboolean('topic', 'topic_on')
  sents_cnt = 0
  words_cnt = 0
  #for lim in range(0, limit):
  pre_scores = 0
  print '[Sen Nums]:',slen
  left_word = max_words
  cused = set([])
  while True:
    max_senid = -1
    max_subfunc = -1
    for i in range(0, slen):
      if i in sel or (max_words > 0 and sens[i].tol_words > left_word):
        continue
      #if sens[i].topic_diversity <= 0.0:
      #  continue
      if sens[i].tol_words < 10:
        continue
      sums.append(sens[i])
      covery, diversity = SubmodularFunction(sums, sens, smatrix, topic_on)
      cc_scores = CalConceptScore(sens[i])
      subfunc = (covery + diversity) #+ 0.2*cc_scores #+ 5 * sens[i].topic_diversity 
      tmpinfo[i] = (subfunc, covery, diversity, cc_scores) 
      if subfunc > max_subfunc:
        max_subfunc = subfunc
        max_senid = i
      del sums[-1]
    if max_senid == -1:
      break
    sums.append(sens[max_senid])
    sel.add(max_senid)
    UpdateConceptWeight(sens[max_senid])
    sents_cnt += 1
    words_cnt += sens[max_senid].sen_len
    left_word -= sens[max_senid].tol_words
    #words_cnt += sens[max_senid].tol_words
    print '[In search]:', max_subfunc, max_subfunc - pre_scores, tmpinfo[max_senid]
    pre_scores = max_subfunc
    if max_sents > 0  and sents_cnt >= max_sents:
      break
    if max_words > 0  and words_cnt >= max_words:
      #del sums[-1]
      #sents_cnt -= 1
      #words_cnt -= sens[max_senid].sen_len
      break
    """
    print 'sel:',max_senid,sens[max_senid].sen_str
    titems = tmpinfo.items()
    slist = sorted(titems, lambda x, y: cmp(x[1][0], y[1][0]), reverse=True) 
    ncnt = 0
    print '======= iteration ========'
    for (key,val) in slist:
      acnt = 0
      bcnt = 0
      for cval in sens[key].con_vector:
        if abs(cval) > 0.1:
          acnt += 1
        else:
          bcnt += 1
      print ncnt,sens[key].sen_id,sens[key].concepts_found,sens[key].topic_diversity,val,sens[key].sen_id, sens[key].word_found, len(sens[key].word_dict),sens[key].sen_str
      ncnt+= 1
      for wrd in sens[key].word_used:
        print wrd,
      print smatrix[0][sens[key].sen_id],sum(smatrix[sens[key].sen_id])
      print '\n'
    """
  print '[summary]:'
  ocnt = 1
  for sen in sums:
    print '[%d] %s diversity:'%(ocnt,sen.sen_order),sen.topic_diversity, 'weight:', sen.weight,'tol_words:', sen.tol_words, 'sen_len:', sen.sen_len, 'rank:', sen.rank_weight
    print sen.sen_str
    ocnt += 1

  print '[GreedySearch] ','sents:',sents_cnt, 'words:',words_cnt
  return sums

def AutoSum(input_file, output_file, vtype='tfidf', w2vec_data=None):
  senlist = ReadText(input_file)
  #senlist = ReadDucText(input_file)
  TfIdfVector(senlist)
  smatrix = None
  if vtype == 'tfidf':
    TfIdfBaseSentenceClustering(senlist)
    smatrix = GetSimMatrix(senlist, 'tfidf')
  elif vtype == 'LSA':
    lsa_dim = int(global_conf.get('config', 'lsa_dim'))
    LSA_Vector(senlist, lsa_dim)
    ConBaseSentenceClustering(senlist)
    smatrix = GetSimMatrix(senlist, 'cos')
  elif vtype == 'w2v':
    w2vec = w2vec_data[0]
    dim = w2vec_data[1]
    Word2Vec_Vector(senlist, w2vec, dim)
    Nsim = (global_conf.get('w2v', 'Nsim') == 'True')
    w2v_similarity = global_conf.get('w2v', 'w2v_similarity')
    #Word2VecReduction(senlist, w2vec, 0.1)
    #TfIdfVector(senlist)
    #TfIdfBaseSentenceClustering(senlist)
    #smatrix = GetSimMatrix(senlist, 'tfidf')
    if Nsim == False:
      #ConBaseSentenceClustering(senlist)
      TfIdfBaseSentenceClustering(senlist)
      #smatrix = GetSimMatrix(senlist, w2v_similarity)
      smatrix = GetSimMatrixWithUnusedWord(senlist, w2v_similarity)
    else:
      #smatrix = GetNSimMatrix(senlist, w2vec_data[0], w2v_similarity)
      smatrix = GetNSimMatrixWithUnusedWord(senlist, w2vec_data[0], w2v_similarity)
      ManiFoldRanking(senlist, w2vec_data[0])
      #SimSentenceClustering(smatrix,senlist)
      #ConBaseSentenceClustering(senlist)
      TfIdfBaseSentenceClustering(senlist)
  elif vtype == 'RAE':
    rae_dir = global_conf.get('rae', 'rae_dir')
    rae_file = rae_dir + input_file.split('/')[-1]
    Load_RAE_vector(rae_file, senlist)
    ConBaseSentenceClustering(senlist)
    smatrix = GetSimMatrix(senlist, 'cos')
    #print smatrix
  res_sum = GreedySearch(senlist, 2, smatrix)
  out_pid = codecs.open(output_file, 'w', 'utf-8')
  for sen in res_sum:
    out_pid.write(sen.sen_str + '\n')
  out_pid.close()
  return res_sum

def FirstNword(input_file, output_file):
  res_sum = []
  senlist = ReadText(input_file)
  N_word = 100
  cnt = 0
  while N_word > 0:
    res_sum.append(senlist[cnt])
    N_word -= senlist[cnt].tol_words
    cnt += 1
    #if cnt >= 2:
    #  break
  #"""
  if N_word >= 0:
    senlist[cnt].sen_str = ' '.join(senlist[cnt].sen_str.split()[:N_word])
    res_sum.append(senlist[cnt])
  #"""
  out_pid = codecs.open(output_file, 'w', 'utf-8')
  for sen in res_sum:
    out_pid.write(sen.sen_str + '\n')
  out_pid.close()
  return res_sum

def ROUGE(sysdoc_dir, modeldoc_dir, prefix_name, vtype='tfidf'):
  r = Rouge155(rouge_args=u"-n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -d")
  #r = Rouge155(rouge_args=u"-a -c 95 -l 100 -m -n 4 -w 1.2")
  #r = Rouge155(rouge_args=u"-a -c 95 -b 665 -m -n 4 -w 1.2")
  #r = Rouge155()
  model_name_suffix = global_conf.get('name_pattern', 'model_name_suffix')
  r.system_dir = sysdoc_dir
  r.model_dir = modeldoc_dir + prefix_name
  r.system_filename_pattern = prefix_name + '.' + vtype + '(\d+).txt'
  r.model_filename_pattern = prefix_name + model_name_suffix #'.(\d).gold'
  output = r.evaluate()
  #print(output)
  lines = output.split('\n')
  for line in lines:
    if line.find(u'Eval 1.1') > 0:
      print global_task_id,line
  output_dict = r.output_to_dict(output)
  return output_dict

def TestROUGE():
  r = Rouge155()
  dirpath = '/home/tuywen/code/summarization/tmp/'
  r.system_dir = dirpath + 'can_doc/'
  r.model_dir = dirpath + 'ref_doc/'
  r.system_filename_pattern = 'a.(\d+).txt'
  r.model_filename_pattern = 'a.(\d+).gold.txt'
  output = r.convert_and_evaluate()
  print(output)
  output_dict = r.output_to_dict(output)
  return output_dict

def ScoreTable(sum_dict, snum):
  keys = sum_dict.keys()
  keys.sort()
  cnt = 0
  for key in keys:
    if key[-2:] != 'cb' and key[-2:] != 'ce':
      if cnt % 3 == 0:
        print
        print 'R-' + key.split('_')[1] + ' \t',
      print "%.3f"%round(sum_dict[key] / snum, 3),
      cnt += 1
  return

def ProcessAllFile(input_dir, output_dir, model_dir, vtype='tfidf'):
  word2vec_file = global_conf.get('config', 'w2v_feature_file')
  w2vec_data= LoadWordVector(word2vec_file)
  run_cases = int(global_conf.get('debug', 'run_cases'))

  dirlist = os.listdir(input_dir)
  dirlist.sort()
  cnt = 0
  sum_dict = {}
  for filename in dirlist:
    input_file = input_dir + filename
    if os.path.isfile(input_file):
      prefix_name = filename.split('.')[0]
      global global_task_id
      global_task_id = prefix_name
      #if prefix_name != 'D0942H-A':
      #  continue
      print '================='*3,cnt,prefix_name,'================='*3
      output_path = output_dir + prefix_name + '/'
      output_file = output_path + prefix_name + '.' + vtype + '001.txt'
      if not os.path.exists(output_path):
        os.mkdir(output_path)
      
      #FindSen(input_file, output_file, vtype, w2vec_data)
      res_sum = []
      if vtype == 'firstnword':
        res_sum = FirstNword(input_file, output_file)
      else:
        res_sum = AutoSum(input_file, output_file, vtype, w2vec_data)
      odict = ROUGE(output_path, model_dir, prefix_name, vtype)
      for k,v in odict.items():
        if k not in sum_dict:
          sum_dict[k] = v
        else:
          sum_dict[k] += v
      print '======================================'*3,'\n'
      cnt += 1
      if run_cases > 0 and cnt >= run_cases:
        break
  ScoreTable(sum_dict, cnt)

  return

def FindSen(input_file, output_file, vtype='tfidf', w2vec_data=None):
  senlist = ReadText(input_file)
  TfIdfVector(senlist)
  tsmatrix = None
  lsmatrix = None
  wsmatrix = None
  
  smatrix = GetSimMatrix(senlist, 'tfidf')
  
  LSA_Vector(senlist, 40)
  lmatrix = GetSimMatrix(senlist, 'cos')
  
  w2vec = w2vec_data[0]
  dim = w2vec_data[1]
  Word2Vec_Vector(senlist, w2vec, dim)
  wmatrix = GetSimMatrix(senlist, 'cos')
  
  slen = len(senlist)
  for i in range(0, slen):
    print '============== %d =============='%(i)
    print 'Sentence: ', senlist[i].sen_str
    titem = [(j, smatrix[i][j]) for j in range(0, slen)]
    stitem= sorted(titem, lambda x,y: cmp(x[1],y[1]), reverse=True)
    tdict = {}
    tcnt = 0
    for (wid, wscore) in stitem:
      tdict[wid] = (tcnt, wscore)
      tcnt += 1

    litem = [(j, lmatrix[i][j]) for j in range(0, slen)]
    slitem= sorted(litem, lambda x,y: cmp(x[1],y[1]), reverse=True)
    ldict = {}
    lcnt = 0
    for (wid, wscore) in slitem:
      ldict[wid] = (lcnt, wscore)
      lcnt += 1
    
    witem = [(j, wmatrix[i][j]) for j in range(0, slen)]
    switem= sorted(witem, lambda x,y: cmp(x[1],y[1]), reverse=True)
    wdict = {}
    wcnt = 0
    for (wid, wscore) in switem:
      wdict[wid] = (wcnt, wscore)
      wcnt += 1
    for j in range(0, slen):
      print stitem[j][1], slitem[j][1], switem[j][1]
    print 
    """
    print '*** TF-IDF ***'
    for j in range(1,6):
      wid = slitem[j][0]
      print tdict[wid],ldict[wid],wdict[wid]
      print senlist[wid].sen_str
      print ''
    """
  return 

def ReadConfigFile(fpath):
  global_conf.read(fpath)

if __name__=='__main__':
  if len(sys.argv) < 1:
    print "print choose config file to run"
  
  ReadConfigFile(sys.argv[1])
  ReadStoplist()
  method = global_conf.get('config', 'method')
  print method
  docset_dir = global_conf.get('config', 'docset_dir')
  sysout_dir = global_conf.get('config', 'sysout_dir')
  goldsum_dir= global_conf.get('config', 'goldsum_dir')

  ProcessAllFile(docset_dir,sysout_dir, goldsum_dir, method)
  #TestROUGE()
