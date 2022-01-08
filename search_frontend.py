from flask import Flask, request, jsonify

######### ENGINE CLASSES ########

# -*- coding: utf-8 -*-
"""after pv.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ogvZRosFRmP4AwiOBqIDQ62LvpWzBtPU

# File path
"""

base_file_path = './'

"""#Imports"""

# Commented out IPython magic to ensure Python compatibility.
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
import math
nltk.download('stopwords')
import builtins
from numpy import dot
from numpy.linalg import norm
from time import time
import numpy as np
import pandas as pd
# %load_ext google.colab.data_table
import bz2
from functools import partial
from collections import Counter, OrderedDict
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import itertools
from time import time
import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

"""#**Inverted Index and writer/reader**"""

import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
from operator import itemgetter
from pathlib import Path
import pickle
from contextlib import closing

BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb') 
                          for i in itertools.count())
        self._f = next(self._file_gen)
    
    def write(self, b):
      locs = []
      while len(b) > 0:
        pos = self._f.tell()
        remaining = BLOCK_SIZE - pos
        # if the current file is full, close and open a new one.
        if remaining == 0:  
          self._f.close()
          self._f = next(self._file_gen)
          pos, remaining = 0, BLOCK_SIZE
        self._f.write(b[:remaining])
        locs.append((self._f.name, pos))
        b = b[remaining:]
      return locs

    def close(self):
      self._f.close()

class MultiFileReader:
  """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
  def __init__(self):
    self._open_files = {}

  def read(self, locs, n_bytes):
    b = []
    for f_name, offset in locs:
      if f_name not in self._open_files:
        self._open_files[f_name] = open(f_name, 'rb')
      f = self._open_files[f_name]
      f.seek(offset)
      n_read = builtins.min(n_bytes, BLOCK_SIZE - offset)
      b.append(f.read(n_read))
      n_bytes -= n_read
    return b''.join(b)
  
  def close(self):
    for f in self._open_files.values():
      f.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
    return False

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

class InvertedIndex:  
  def __init__(self, docs={}):
    """ Initializes the inverted index and add documents to it (if provided).
    Parameters:
    -----------
      docs: dict mapping doc_id to list of tokens
    """
    # stores document frequency per term
    self.df = Counter()
    # stores total frequency per term
    self.term_total = Counter()
    # stores posting list per term while building the index (internally), 
    # otherwise too big to store in memory.
    self._posting_list = defaultdict(list)
    # mapping a term to posting file locations, which is a list of 
    # (file_name, offset) pairs. Since posting lists are big we are going to
    # write them to disk and just save their location in this list. We are 
    # using the MultiFileWriter helper class to write fixed-size files and store
    # for each term/posting list its list of locations. The offset represents 
    # the number of bytes from the beginning of the file where the posting list
    # starts. 
    self.posting_locs = defaultdict(list)


    
    for doc_id, tokens in docs.items():
      self.add_doc(doc_id, tokens)

  def add_doc(self, doc_id, tokens):
    """ Adds a document to the index with a given `doc_id` and tokens. It counts
        the tf of tokens, then update the index (in memory, no storage 
        side-effects).
    """
    w2cnt = Counter(tokens)
    self.term_total.update(w2cnt)
    for w, cnt in w2cnt.items():
      self.df[w] = self.df.get(w, 0) + 1
      self._posting_list[w].append((doc_id, cnt))

  def write_index(self, base_dir, name):
    """ Write the in-memory index to disk. Results in the file: 
        (1) `name`.pkl containing the global term stats (e.g. df).
    """
    self._write_globals(base_dir, name)

  def _write_globals(self, base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
      pickle.dump(self, f)

  def __getstate__(self):
    """ Modify how the object is pickled by removing the internal posting lists
        from the object's state dictionary. 
    """
    state = self.__dict__.copy()
    del state['_posting_list']
    return state

  def posting_lists_iter(self):
    """ A generator that reads one posting list from disk and yields 
        a (word:str, [(doc_id:int, tf:int), ...]) tuple.
    """
    with closing(MultiFileReader()) as reader:
      for w, locs in self.posting_locs.items():
        b = reader.read(locs, self.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(self.df[w]):
          doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
          tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
          posting_list.append((doc_id, tf))
        yield w, posting_list


  @staticmethod
  def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
      return pickle.load(f)

  @staticmethod
  def delete_index(base_dir, name):
    path_globals = Path(base_dir) / f'{name}.pkl'
    path_globals.unlink()
    for p in Path(base_dir).rglob(f'{name}_*.bin'):
      p.unlink()


  @staticmethod
  def write_a_posting_list(b_w_pl,location):
    ''' Takes a (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...]) 
    and writes it out to disk as files named {bucket_id}_XXX.bin under the 
    current directory. Returns a posting locations dictionary that maps each 
    word to the list of files and offsets that contain its posting list.
    Parameters:
    -----------
      b_w_pl: tuple
        Containing a bucket id and all (word, posting list) pairs in that bucket
        (bucket_id, [(w0, posting_list_0), (w1, posting_list_1), ...])
    Return:
      posting_locs: dict
        Posting locations for each of the words written out in this bucket.
    '''
    posting_locs = defaultdict(list)
    bucket, list_w_pl = b_w_pl

    with closing(MultiFileWriter(location, bucket)) as writer:
      for w, pl in list_w_pl: 
        # convert to bytes
        b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
      # save file locations to index
        posting_locs[w].extend(locs)
    return posting_locs
def find_postings(terms,index):
    res={}
    with closing(MultiFileReader()) as reader:
        for w, locs in index.posting_locs.items():
          if(res.keys()==terms):
          #if (terms in res.keys()):
            break
          if(w not in terms):
            continue
          else:
            b = reader.read(locs, index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[w]):
              doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
              tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
              posting_list.append((doc_id, tf))
            res[w]=posting_list

    return res
"""Authenticate """

# Authenticate your user
# The authentication should be done with the email connected to your GCP account
#from google.colab import auth
#auth.authenticate_user()

"""#tf, df, tokenizing and other helper functions"""

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", #TODO: CHECK IF NEED TO ADD
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
#Flattens a list of lists to one single list
def flatten_lst(t):
    return [item for sublist in t for item in sublist]

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return tokens

def tokenize_list_of_texts(lst):
    output = []
    for text in lst:
      tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
      output.append(tokens)
    return output
# Calc TF
# Returns a list of (token, (doc_id, tf)) pairs for each token (word) in the text
def word_count(id,text):
  ''' Count the frequency of each word in `text` (tf) that is not included in 
  `all_stopwords` and return entries that will go into our posting lists. 
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs 
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
 
  terms = {}
  tf = Counter()
  for token in tokens:
    if token in all_stopwords:
      continue
    current = tf.get(token, 0)
    tf[token] = current + 1
    terms[token] = (id,current +1)


  return list(terms.items())
def word_count_anchor(id,list_of_tupels):
  ''' Count the frequency of each word in `text` (tf) that is not included in 
  `all_stopwords` and return entries that will go into our posting lists. 
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs 
      for example: [("Anarchism", (12, 5)), ...]
  '''
  t=flatten_lst([RE_WORD.finditer(text.lower()) for dest_id,text in list_of_tupels])
  tokens = [token.group() for token in t]

  terms = {}
  tf = Counter()
  for token in tokens:
    if token in all_stopwords:
      continue
    current = tf.get(token, 0)
    tf[token] = current + 1
    terms[token] = (id,current +1)

  return list(terms.items())
# Returns a sorted posting list from an unsorted posting list. Sorting by tf of doc id
def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples 
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''

  return(sorted(unsorted_pl, key = lambda x: x[0]))

def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
  # YOUR CODE HERE
  return postings.map(lambda x: (x[0],len(x[1]))).partitionBy(16)
def partition_postings_and_write(postings,location):
  ''' A function that partitions the posting lists into buckets, writes out 
  all posting lists in a bucket to disk, and returns the posting locations for 
  each bucket. Partitioning should be done through the use of `token2bucket` 
  above. Writing to disk should use the function  `write_a_posting_list`, a 
  static method implemented in inverted_index_colab.py under the InvertedIndex 
  class. 
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and 
      offsets its posting list was written to. See `write_a_posting_list` for 
      more details.
  '''
  res = defaultdict(list)
  output_list = []
  bucketed_postings = postings.map(lambda x:(token2bucket_id(x[0]),x)).groupByKey()
  for id,content in bucketed_postings.toLocalIterator():
    output_list.append(InvertedIndex.write_a_posting_list((id,content),location))
  sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
  return sc.parallelize(output_list,numSlices=16)
"""# loading big index and DLs : <font color='green'>Indexes here</font>

loading DLs
"""

import json
with open(base_file_path+"title_index/"+'title_DL.json', 'rt') as f:
  title_DL = json.load(f)
#with open(base_file_path+"anchor_index/"+'dl_anchor_new.json', 'rt') as f:
#  anchor_DL = json.load(f)
with open(base_file_path+"body_index/"+'body_DL.json', 'rt') as f:
  body_DL = json.load(f)

"""loading indices"""

title_index=InvertedIndex.read_index(base_file_path+'title_index', 'index')
anchor_index=InvertedIndex.read_index(base_file_path+'anchor_index', 'index')
body_index=InvertedIndex.read_index(base_file_path+'body_index', 'index')
#!cp base_file_path+'body_index'

"""Changing posting locs to match local colab file location"""

def change_index_locs(index):
    for term,loc_tuple in index.posting_locs.items():
      new_list = []
      for loc,offset in loc_tuple:
        splited = loc.split("/")
        for part in splited:
          if "index" in part:
            rebuilt = part
          if "bin" in part:
            rebuilt = base_file_path + rebuilt + "/" + part
        loc_tuple = (rebuilt,offset)
        new_list.append(loc_tuple)
      index.posting_locs[term] = new_list
if True: #Change to "True" to enable chaging the posting locs in the indices to Mydrive location
  change_index_locs(body_index)
  change_index_locs(anchor_index)
  change_index_locs(title_index)
def find_postings(terms,index):
    res={}
    with closing(MultiFileReader()) as reader:
        for w, locs in index.posting_locs.items():
          if(res.keys()==terms):
          #if (terms in res.keys()):
            break
          if(w not in terms):
            continue
          else:
            b = reader.read(locs, index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[w]):
              doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
              tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
              posting_list.append((doc_id, tf))
            res[w]=posting_list

    return res
"""# TF-idf and cosine similarity"""

def tf_idf(t,d,DL,index,tf):
  term_freq_not_normelized=tf
  D=DL[str(d)]
  tf=term_freq_not_normelized/D
  N=len(DL)
  idf=math.log2(N/index.df[t])
  return tf*idf
def cosine_sim(index,DL,query,N = 3):
  dict_count = Counter()
  pls = find_postings(np.unique(query).tolist(),index)
  tf_q = Counter()
  d_tfs = defaultdict(list)
  for t in query:
    tf_q[t] += 1
  for t,pl in pls.items():
    for doc_id,tf_d in pl:
      tf_idf_val=tf_idf(t,doc_id,DL,index,tf_d)
      dict_count[doc_id] += (tf_q[t]) * tf_idf_val
      d_tfs[doc_id].append(tf_idf_val)
  for doc_id,value in dict_count.items():
    dict_count[doc_id] = value*(1/(norm(list(tf_q.values()))))*(1/(norm(list(d_tfs[doc_id]))))
  return sorted([(doc_id, builtins.round(score,5)) for doc_id, score in dict_count.items()], key = lambda x: x[1],reverse=True)[:N]
def getCosineSim(query):
    res = cosine_sim(body_index,body_DL,query,N=100)
    output = []
    doc_list =  [doc_id for doc_id,rank in res]
    for doc in doc_list:
      output.append((doc,getTitle(doc)))
    return output

"""# binary ranking using the title of articles and anchor text


"""
def binary_ranking_title_and_anchor_text(query,index):
  matches = Counter()
  pls = find_postings(np.unique(query).tolist(),index)
  for term in np.unique(query): 
      list_of_doc = pls[term.lower()]    
      for doc_id, freq in list_of_doc:
          matches[doc_id] += freq
  return sorted(matches.items(),key=lambda x:x[1], reverse =True)

def binary_ranking_title(query):
    ranking = binary_ranking_title_and_anchor_text(query,title_index)
    output = []
    doc_list = [doc_id for doc_id,freq in ranking]
    for doc in doc_list:
      output.append((doc,getTitle(doc)))
    return output

def binary_ranking_anchor_text(query):
    ranking = binary_ranking_title_and_anchor_text(query,anchor_index)
    output = []
    doc_list =  [doc_id for doc_id,freq in ranking]
    for doc in doc_list:
      output.append((doc,getTitle(doc)))
    return output

"""# ranking by PageRank """

def getPr(id):
  k=id%800
  with open(base_file_path+'page_rank/pr'+str(k)+'.json', 'rt') as f:
     pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

def getPageRank(id_list):
    output = []
    for id in id_list:
      output.append(getPr(id))
    return output

"""# ranking by article page views """

def getPv(id):
  k=id%800
  with open(base_file_path+'pv/pv'+str(k)+'.json', 'rt') as f:
    pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

def getPageViews(id_list):
    output = []
    for id in id_list:
      output.append(getPv(id))
    return output

"""# BM25 helper function"""
def get_candidate_documents(query_to_search,index,pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. 
    
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    pls: relevent pls for query terms
    Returns:
    -----------
    list of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score. 
    """
    candidates = []    
    for term,pl in pls.items():
        if term in index.df.keys():        
            current_list = pls[term.lower()]    
            candidates += current_list  

    return np.unique([d for d,s in candidates])

"""# BM25 class"""

import math
from itertools import chain
import time
# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self,index,DL,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = builtins.sum(DL.values())/self.N
        self.DL = DL

    def calc_idf(self,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """        
        idf = {}        
        for term in list_of_tokens:            
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass                             
        return idf
        

    def search(self, queries,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query. 
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """
        terms = []
        for query in queries:
          for term in query:
            terms.append(term)
        uniques = set(terms)
        self.idf = self.calc_idf(uniques)
        output = OrderedDict()
        for i,query in enumerate(queries):
            scores = []
            #loading pls for the specific query's terms
            pls = find_postings(np.unique(query).tolist(),self.index)
            candidates = get_candidate_documents(query,self.index,pls)
            scores = list(self._score(query, candidates,pls).items())
            scores.sort(reverse = True,key = lambda x:x[1])
            output[i] = scores[:N]
        return output

    def _score(self, query, docs,pls):
        """
        This function calculate the bm25 score for given query and document.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """        
        scores = Counter() 

        for term in query:
            if term in self.index.df.keys():            
                term_frequencies = dict(pls[term.lower()])
                relevent_docs =  intersection(docs,[doc_id for doc_id,tf in term_frequencies.items()])
                for doc_id in relevent_docs:
                    tf = term_frequencies[doc_id]
                    numerator = self.idf[term] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.DL[str(doc_id)] / self.AVGDL)
                    scores[doc_id] += (numerator / denominator)
        return scores

"""# BM25 merge results"""

def merge_results(title_scores,body_scores,title_weight=0.58,text_weight=0.43,N = 100):    
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body). 

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
                
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function. 
    
    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score). 
    """
    output = OrderedDict()
    for key in title_scores:
      temp = []
      title_values = title_scores[key]
      body_values = body_scores[key]
      for id,title in title_values:
        for id2,body in body_values:
          if id == id2:
            temp.append((id,title*title_weight+body*text_weight))
        if id not in (item[0] for item in temp):
          temp.append((id,title*title_weight))
      for id,body in body_values:
        if id not in (item[0] for item in temp):
          temp.append((id,body*text_weight))
      temp.sort(key = lambda x:x[1], reverse = True)
      output[key] = temp[:N]
    return output

def merge_results_for_three(title_scores,body_scores,anchor_scores,title_weight=0.25,text_weight=0.5,anchor_weight=0.25,N = 40):
    merged_title_body = merge_results(title_scores,body_scores,title_weight,text_weight,500)
    return merge_results(merged_title_body,anchor_scores,1,anchor_weight,N)

"""# BM25 functionallity functions"""
#Create BM25 instances for body, title and anchor text.
bm25_body = BM25_from_index(body_index,body_DL)
bm25_title = BM25_from_index(title_index,title_DL)
#function to run bm25 on the given queries and get merged results from bm25 on body of the docs and title of the docs
def merged_BM25_for_queries(queries_to_test,title_weight=0.574257,text_weight=0.425742,N = 100):
    body_bm25_res = bm25_body.search(queries_to_test,N=N)
    title_bm25_res = bm25_title.search(queries_to_test,N=N)
    merged_title_body = merge_results(title_bm25_res,body_bm25_res,title_weight=title_weight,text_weight=text_weight,N=N)
    return merged_title_body

def getTitle(id):
  k=id%800
  with open('./doc2title/d2t'+str(k)+'.json', 'rt') as f:
     pr_k = json.load(f)
  for t in pr_k:
    if t[0]==id:
      return t[1]
  return 0

"""# Estimate



"""

#@title Default title text
def intersection(l1,l2):      
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(builtins.set(l1)&builtins.set(l2))
def precision_at_k(true_list,predicted_list,k=40):    
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list
    
    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """      
    # YOUR CODE HERE

    up=len(intersection(predicted_list[:k],true_list))
    down=k
    return builtins.round(up/down,3)

def average_precision(true_list,predicted_list,k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).     

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list
    
    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    # for each relevant doc that in pl and in tl activate p@k and sum them,after 
    #devide by number of relevant
    sum=0
    rel=len(intersection(predicted_list[:k],true_list))
    if rel==0:
      return 0
    for i_pl,doc in enumerate(predicted_list[:k]):
      if doc in true_list:
        sum+=precision_at_k(true_list,predicted_list[:k],i_pl+1)
    return builtins.round(sum/rel,3)

def mean_ap(true_dict,predicted_dict,k=40):
  sum=0
  for q_id,true_lst in true_dict.items():
    sum+=average_precision(true_lst,predicted_dict[q_id])
  return sum/len(true_dict)

"""# Main Engine Function"""

def search_engine(query,N=100):
    merged = merged_BM25_for_queries(query,title_weight=0.11607,text_weight=0.88392,N=N)
    doc_list =  [t[0] for t in [lst for qid,lst in merged.items()][0]]
    output = []
    for doc in doc_list:
      output.append((doc,getTitle(doc)))
    return output

##TESTS FOR WEIGHTS ####################################################



def merged_BM25_for_queries_tests(index,title_weight=0.5,text_weight=0.65,N = 100):
    merged_title_body = merge_results(bm_title_dict[index],bm_body_dict[index],title_weight=title_weight,text_weight=text_weight,N=N)
    return merged_title_body


def test_diff_weights_4merge_two(bm25_body,bm25_title,increments = 5):
      x = 0
      y = 0
      max=0
      for x in range(0,100,increments):
        for y in range(0,100,increments):
            res_dict = {}
            t_start = time.time()
            for index,query in enumerate(queries):
              merged = merged_BM25_for_queries_tests(index,title_weight=x/100,text_weight=y/100,N=100)
              res_dict[index] = [t[0] for t in [lst for qid,lst in merged.items()][0]]

            cur = mean_ap(qid_to_true_docs_dict,res_dict,40)
            if cur > max:
              max = cur
              best = (x,y)
              print(max,best)
              print(res_dict[0])
print("b4 queries load")
import json
with open('queries_train.json', 'rt') as f:
  queries_to_docs_raw = json.load(f)
print("b4 queries functions")
doc_lst=flatten_lst([docs for q,docs in queries_to_docs_raw.items()])
queries=[q for q,d in queries_to_docs_raw.items()]
q_tokenes_lst=tokenize_list_of_texts(queries)
qid_to_true_docs_dict={i:d[1] for i,d in enumerate(queries_to_docs_raw.items())}
print("b4 test bm25 searches")
# body_bm25_res = bm25_body.search(q_tokenes_lst)
# title_bm25_res = bm25_title.search(q_tokenes_lst)
bm_body_dict = {}
bm_title_dict = {}
if False:
  for index,query in enumerate(queries):
      tokenized_query = tokenize(query)
      bm_body_dict[index] = bm25_body.search([tokenized_query])
      bm_title_dict[index] = bm25_title.search([tokenized_query])
  test_diff_weights_4merge_two(bm25_body,bm25_title,increments = 1)


##END OF TESTS FOR WEIGHTS ####################################################

######### END ENGINE CLASSES ########


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    t_start = time.time()
    tokenized_query = tokenize(query)
    res = search_engine([tokenized_query],N=100)
    print("Time for query",time.time() - t_start)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = getCosineSim(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = binary_ranking_title(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokenized_query = tokenize(query)
    res = binary_ranking_anchor_text(tokenized_query)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = getPageRank(wiki_ids)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = getPageViews(wiki_ids)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
