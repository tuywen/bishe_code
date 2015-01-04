#!/bin/sh
log_dir=$1
nohup python data_preprocess_n.py configure/duc2010.conifg.tfidf > ${log_dir}/duc10.tfidf 2>&1 &
nohup python data_preprocess_n.py configure/duc2010.conifg.LSA   > ${log_dir}/duc10.LSA 2>&1 &
nohup python data_preprocess_n.py configure/duc2010.conifg.w2v.euc   > ${log_dir}/duc10.w2v.euc 2>&1 &
nohup python data_preprocess_n.py configure/duc2010.conifg.w2v.cos > ${log_dir}/duc10.w2v.cos 2>&1 &
nohup python data_preprocess_n.py configure/duc2010.conifg.w2v.Nsim > ${log_dir}/duc10.w2v.Nsim 2>&1 &
nohup python data_preprocess_n.py configure/duc2010.conifg.w2v.Nsim.match > ${log_dir}/duc10.w2v.Nsim.match 2>&1 &
