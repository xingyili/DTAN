# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 5e-5 --im_epoch 25


# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 5e-5 --im_epoch 25


# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 1.0 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 1.0 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low ae --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 1.0 --im_lambda 5e-5 --im_epoch 25
















# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 0.33333333 --im_lambda 5e-5 --im_epoch 25


# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 0.66666666 --im_lambda 5e-5 --im_epoch 25


# -------------------------------- H1 --------------------------------
python main_time.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 1.0 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H3 --------------------------------
python main_time.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64 --epoch 20 --model full --device cuda:3 --percentage 1.0 --im_lambda 3e-5 --im_epoch 10

# -------------------------------- H5 --------------------------------
python main_time.py --seq_path data/dataset0722/H5_HA_syed.csv --anti_path data/dataset0722/H5_HI_syed.csv --subtype H5 --label NHT_distance --valid_type titer --n_jobs 30  --low none --ae_dim 64  --epoch 300 --model full --device cuda:3 --percentage 1.0 --im_lambda 5e-5 --im_epoch 25