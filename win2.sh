# -------------------------------- H3 --------------------------------
python main.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model cnn --im_lambda 3e-5 --im_epoch 10 --device cuda:4


# -------------------------------- H3 --------------------------------
python main.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model no_encoder --im_lambda 3e-5 --im_epoch 10 --device cuda:4


# -------------------------------- H3 --------------------------------
python main.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model no_meta --im_lambda 3e-5 --im_epoch 10 --device cuda:4

# -------------------------------- H3 --------------------------------
python main.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low onehot --ae_dim 20 --epoch 20 --model full --im_lambda 3e-5 --im_epoch 10 --device cuda:4