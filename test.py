import sys
import os
import codecs

def add(a,b):
  alen = len(a)
  for i in range(0, alen):
    a[i] += b[i]

if __name__=='__main__':
  a=[0,0]
  add(a,[1,1])
  print a
  add(a,[2,3])
  print a
