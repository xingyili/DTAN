# -------------------------------- H1 --------------------------------
python main.py --seq_path data/dataset0722/H1N1_N2_HA_syed.csv --anti_path data/dataset0722/H1N1_N2_HI_syed.csv --subtype H1 --label NHT_distance --valid_type test --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --im_lambda 3e-5 --im_epoch 10 --device cuda:3 --test_file download_ret/H1_with_seq.csv

# -------------------------------- H3 --------------------------------
python main.py --seq_path data/dataset0722/H3N2_full_HA_syed.csv --anti_path data/dataset0722/H3N2_full_HI_syed.csv --subtype H3 --label NHT_distance --valid_type test --n_jobs 30  --low ae --ae_dim 64 --epoch 20 --model full --im_lambda 3e-5 --im_epoch 10 --device cuda:4 --test_file download_ret/H3_with_seq.csv