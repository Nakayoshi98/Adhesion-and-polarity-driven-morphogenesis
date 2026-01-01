for seed in `seq 1 10`
do
for p in 7 #`seq 0 10` 
do
for tau_B in 3 #`seq 0 15` 
do
./model_base_2D $seed $p $tau_B &&
python DataProcessing/visualize_2D.py $seed $p $tau_B &
done 
done
done
