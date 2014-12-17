from __future__ import division
import sys
import nltk
import os
import codecs

def ReadAndCut(input_file):
  infile = codecs.open(input_file, 'r', 'utf-8')
  raw_text = infile.read()
  infile.close()
  tokens = nltk.word_tokenize(raw_text.lower())
  return tokens

def ProcessAllFile(input_dir, output_file):
  dirlist = os.listdir(input_dir)
  cnt = 0
  sum_dict = {}
  for filename in dirlist:
    input_file = input_dir + filename
    if os.path.isfile(input_file):
      prefix_name = filename.split('.')[0]
      print cnt,prefix_name
      tokens = ReadAndCut(input_file)
      for token in tokens:
        if token not in sum_dict:
          sum_dict[token] = 1
        else:
          sum_dict[token] += 1
      cnt += 1
      #break
  ofile  = codecs.open(output_file, 'w', 'utf-8')
  sitems = sum_dict.items()
  slist = sorted(sitems, lambda x, y: cmp(x[1], y[1]), reverse=True)
  for (key,val) in slist:
    ofile.write(key + ' ' + str(val) + '\n')
  ofile.close()
  return

if __name__=='__main__':
  cdir = '/home/tuywen/code/summarization/data/'
  ProcessAllFile(cdir + 'OpinosisDataset1.0_0/topics_2/','/home/tuywen/code/summarization/tmp/all_words.txt')
