import sys
from nltk import word_tokenize

def statfile(fname):
  infile = open(fname, 'r')
  wdict = {}
  for line in infile:
    topic_str = ' '.join(line.strip().split('####')[1:])
    words = word_tokenize(topic_str.lower())
    for w in words:
      if w not in wdict:
        wdict[w] = 1
      else:
        wdict[w] += 1
  
  witems = wdict.items()
  sort_list = sorted(witems, key=lambda x:x[1], reverse=True)
  for sitem in sort_list:
    print sitem[0], sitem[1]
  return

if __name__=='__main__':
  statfile(sys.argv[1])
