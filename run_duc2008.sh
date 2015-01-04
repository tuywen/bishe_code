#!/bin/sh
nohup python data_preprocess_n.py configure/duc2008.conifg.tfidf > log3/duc08.tfidf 2>&1 &
nohup python data_preprocess_n.py configure/duc2008.conifg.LSA   > log3/duc08.LSA 2>&1 &
nohup python data_preprocess_n.py configure/duc2008.conifg.w2v.euc   > log3/duc08.w2v.euc 2>&1 &
nohup python data_preprocess_n.py configure/duc2008.conifg.w2v.cos > log3/duc08.w2v.cos 2>&1 &
