#!/bin/sh

#python stat_word.py ../data/duc2008/topic/ ../tmp/duc2008_words.txt
#python stat_word.py ../data/duc2009/topic/ ../tmp/duc2009_words.txt 
#python stat_word.py ../data/duc2010/topic/ ../tmp/duc2010_words.txt 

#w2v_file="/home/tuywen/code/summarization/tools/w2vec/out_words/GoogleNews-vectors-negative300u.txt"
#out_dir="/home/tuywen/code/summarization/tools/w2vec/out_words/"
#python filter.py ../tmp/duc2008_words.txt ${w2v_file} ${out_dir}duc2008.w2v
#python filter.py ../tmp/duc2009_words.txt ${w2v_file} ${out_dir}duc2009.w2v
#python filter.py ../tmp/duc2010_words.txt ${w2v_file} ${out_dir}duc2010.w2v

cw_file="/home/tuywen/code/summarization/tools/CW_vec/CW_scaled_200.txt"
out_dir="/home/tuywen/code/summarization/tools/CW_vec/"
python filter.py ../tmp/duc2008_words.txt ${cw_file} ${out_dir}duc2008.cw200
python filter.py ../tmp/duc2009_words.txt ${cw_file} ${out_dir}duc2009.cw200
python filter.py ../tmp/duc2010_words.txt ${cw_file} ${out_dir}duc2010.cw200
