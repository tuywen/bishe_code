cat $1 | grep "1 ROUGE-1 Eval 1.1" | awk 'BEGIN{OFS="\t"}{split($8,a,":");print $1,a[2]}'
