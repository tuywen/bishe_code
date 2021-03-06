[config]
docset_dir = /home/tuywen/code/summarization/data/duc2008/topic/
goldsum_dir = /home/tuywen/code/summarization/data/duc2008/gold/
sysout_dir = /home/tuywen/code/summarization/duc_out/duc2008/
method = w2v
lsa_dim = 100
w2v_feature_file= /home/tuywen/code/summarization/tools/CW_vec/duc2008.cw200
w2v_similarity= cos 
similarity= cos

[debug]
run_cases = -1

[name_pattern]
model_name_suffix = .M.100.[A-Z]

[greedysearch]
max_sents = -1
max_words = 100

[w2v]
Nsim = False
