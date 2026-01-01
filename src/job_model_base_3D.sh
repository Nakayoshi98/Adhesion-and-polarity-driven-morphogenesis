for seed in 0 #`seq 1 10`
do
for p in 7 #`seq 1 15` 
do
for tau_B in 12 #`seq 1 20`
do
./model_base_3D $seed $p $tau_B &&
python DataProcessing/visualize_3D.py $seed $p $tau_B &
done 
done
done
