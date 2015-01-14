import sys
import os
from xml.etree import ElementTree
from nltk.tokenize import sent_tokenize

def GetNodeSents(root, nodename, hdate='20140101'):
  nodes = root.getiterator(nodename)
  sents = []
  pcnt = 0
  for node in nodes:
    pcnt += 1
    text = node.text.strip()
    text = text.replace('\n', ' ')
    tmp_sents = sent_tokenize(text)
    scnt = 0
    for st in tmp_sents:
        scnt += 1
        if nodename == 'HEADLINE':
            sents.append('H#@#%s#@#%d#@#%s'%(hdate,scnt,st))
        else:
            sents.append('P#@#%d#@#%d#@#%s'%(pcnt,scnt,st))
  return sents

def GetDate(datestr):
    dstr = datestr.split('.')[0]
    return dstr[-8:]

def ChangeXML(dirname, opath):
  files = os.listdir(dirname)
  sents = []
  ssum = 0
  for filename in files:
    print filename
    fpath = dirname + filename
    tmpstr = (open(fpath, 'r')).read().replace('&',' ')
    #root = ElementTree.parse(fpath)
    root = ElementTree.fromstring(tmpstr)
    sents += GetNodeSents(root, 'HEADLINE', GetDate(filename))
    text_node = root.getiterator('TEXT')
    pnode_num = len(text_node[0].getchildren())
    if pnode_num > 0:
      sents += GetNodeSents(root, 'P')
    else:
      sents += GetNodeSents(root, 'TEXT')
  
  scnt = 0  
  ofile = open(opath, 'w')
  for s in sents:
    #print scnt,s
    ofile.write(s + '\n')
    scnt += 1
  ofile.close()

def ProcessDir(fpath,opath):
  dirs = os.listdir(fpath)
  for adir in dirs:
    print adir
    ofile = opath + adir + '-A'
    ChangeXML(fpath+adir+'/'+adir+'-A/', ofile)
    #break
  return

def ChangeGold(ifile, ofile):
  infile = open(ifile, 'r')
  outfile = open(ofile, 'w')
  for line in infile:
    outfile.write(line)
  infile.close()
  outfile.close()
  return

def ProcessGoldSummaries(inpath, outpath):
  files = os.listdir(inpath)
  for filename in files:
    vals = filename.split('.')
    tvals = vals[0].split('-')
    topic = tvals[0]
    docset = tvals[1]
    selector = vals[3]
    sumid = vals[4]
    if docset == 'B':
      continue
    outdir = outpath + "%s%s-%s/"%(topic, selector, docset)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    outfile = outdir + "%s%s-%s.M.100.%s"%(topic, selector, docset, sumid)
    infile = inpath + filename
    ChangeGold(infile, outfile)
    print filename
    #break
  return

if __name__=='__main__':
  reload(sys)
  sys.setdefaultencoding('utf-8')
  """
  f='/home/tuywen/MyShare/bishe/DUC_data/duc08/UpdateSumm08_test_docs_files/D0801A/D0801A-A/AFP_ENG_20050115.0485'
  f='/home/tuywen/MyShare/bishe/DUC_data/duc08/UpdateSumm08_test_docs_files/D0801A/D0801A-A/AFP_ENG_20050116.0346'
  fpath='/home/tuywen/MyShare/bishe/DUC_data/duc08/UpdateSumm08_test_docs_files/'
  doc_outpath = '/home/tuywen/code/summarization/data/duc2008/topic/'
  #ReadXML(f)
  ProcessDir(fpath, doc_outpath)

  gold_dir = '/home/tuywen/MyShare/bishe/DUC_data/duc08/UpdateSumm08_eval/ROUGE/models/'
  gold_out = '/home/tuywen/code/summarization/data/duc2008/gold/'
  ProcessGoldSummaries(gold_dir, gold_out)
  """
  if len(sys.argv) < 3:
    print 'Usage: python duc_data.py doc_dir model_dir dest_dir'
  else:
    doc_dir = sys.argv[1]
    model_dir = sys.argv[2]
    dest_dir = sys.argv[3]
    os.system('rm -rf %s'%(dest_dir+'topic2/*'))
    os.system('rm -rf %s'%(dest_dir+'gold2/*'))
    ProcessDir(doc_dir, dest_dir+'topic2/')
    ProcessGoldSummaries(model_dir, dest_dir+'gold2/')
