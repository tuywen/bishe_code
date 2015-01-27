params=(0.05 0.06 0.08 0.09 0.1 0.12 0.14 0.16 0.18 0.2 0.3 0.4)
log_dir="grid_log/"

cnt=0
for p in ${params[@]}
do
  echo $p
  nohup python data_preprocess_n.py configure/duc2009.conifg.best $p > ${log_dir}log_${p} 2>&1 &
  ((cnt++))
  d=$((cnt%4))
  #echo $cnt, $d
  if [ $d -eq 0 ];then
    sleep 10m
  fi
done
