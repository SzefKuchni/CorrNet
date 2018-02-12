
for i in {2..4}
do
  let j=$i+1
  while [[ 5 -gt $j ]]
  do
    mkdir output/ABCD/T_$i$j
    python train_corrnet4.py input/M_ABCD/ output/ABCD/T_$i$j/
    let j=$j+1
  done
done
