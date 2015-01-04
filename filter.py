#!/usr/bin/python
import sys

def ReadDict(path):
  odict = {}
  infile = open(path, 'r')
  for line in infile:
    skey = line.strip().split()[0]
    skey = skey.lower()
    if skey not in odict:
      odict[skey] = 0
  return odict

def ReadVector(path, opath, odict):
  infile = open(path, 'r')
  ofile = open(opath, 'w')
  not_find = open('not_find.txt','w')
  cnt = 1
  for line in infile:
    vals = line.strip().split()
    vkey = vals[0].lower()
    if vkey in odict:
      if odict[vkey] == 0:
        ofile.write(line)
        odict[vkey] = 1
    print 'line: %d\r'%(cnt),
    cnt += 1
  for key in odict:
    if odict[key] == 0:
      not_find.write(key + '\n')
  infile.close()
  ofile.close()
  not_find.close()

if __name__=='__main__':
  if len(sys.argv) >= 4:
    oset = ReadDict(sys.argv[1])
    ReadVector(sys.argv[2], sys.argv[3], oset)
  else:
    print 'Usage: python filter.py word_dict w2v_file outfile'
