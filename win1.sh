# -------------------------------- H1 --------------------------------
python main.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model cnn --im_lambda 3e-5 --im_epoch 10 --device cuda:3 


########################### noenc
python main.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model no_encoder --im_lambda 3e-5 --im_epoch 10 --device cuda:3 


########################### nometa
python main.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model no_meta --im_lambda 3e-5 --im_epoch 10 --device cuda:3 

########################### onehot
# -------------------------------- H1 --------------------------------
python main.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low onehot --ae_dim 20 --epoch 20 --model full --im_lambda 3e-5 --im_epoch 10 --device cuda:3 

