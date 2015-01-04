#!/bin/sh

rae_path=/home/tuywen/code/summarization/RAE_code/
tools=/home/tuywen/code/summarization/tools/codeRAEVectorsNIPS2011/
topic_dir=$1
ls ${topic_dir} > tmp_code.txt
while read line
do
  #input_file=$1
  #fname=`basename ${input_file}`
  fname=$line
  #echo $fname

  out_file=${rae_path}${fname}
  echo $out_file

  if [ ! -f "$out_file" ];then
    cd ${tools}
    ./duc_phrase2Vector.sh ${topic_dir}${fname}
    cp outVectors.txt ${out_file}
    cd -
  fi
done < tmp_code.txt
