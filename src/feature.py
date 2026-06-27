import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from aaindex import aaindex1
from src.model import Autoencoder, DenoisingAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm
import joblib
import os

class AeFeature:
    def __init__(self, args):
        self.n_jobs = args.n_jobs
        self.aaidnex1_record_codes = aaindex1.record_codes()
        self.aaindex1_dict = {code: aaindex1[code].values for code in self.aaidnex1_record_codes}
        self.AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
        self.origin_embedding = self.init_embedding()
        self.ae_dim = args.ae_dim
        
        if args.low == "ae":
            self.embedding = self.encode_aa()
        elif args.low == "pca":
            self.embedding = self.encode_pca()
        elif args.low == "onehot":
            self.embedding = self.encode_onehot()
        else:
            scaler = StandardScaler()
            sc_embedding = scaler.fit_transform(self.origin_embedding)
            self.embedding = sc_embedding
            self.ae_dim = 566

    def init_embedding(self):
        origin_embedding  = np.empty((20, 566), dtype=np.float32)
        for aa in self.AA_LIST:
            for record in self.aaidnex1_record_codes:
                origin_embedding[self.AA_LIST.index(aa), self.aaidnex1_record_codes.index(record)] = self.aaindex1_dict[record][aa]
        return origin_embedding

    def encode_aa(self):
        scaler = StandardScaler()
        sc_embedding = scaler.fit_transform(self.origin_embedding)
        # sc_embedding = self.origin_embedding
        ts_embedding = torch.tensor(sc_embedding, dtype=torch.float32)
        model_cache = f"metadata/ae_model_{self.ae_dim}.joblib"
        if os.path.exists(model_cache) == False:
            model = DenoisingAutoencoder(566, 200, self.ae_dim)
            # model = Autoencoder(566, 200, self.ae_dim)
            # model = VAE(566, 200, self.ae_dim)
            epochs = 1500
            lr = 1e-3
            weight_decay = 1e-4
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            best_loss = float('inf')
            best_model_state = None
            
            pbar = tqdm.tqdm(range(1, epochs + 1), miniters=10)
            for epoch in pbar:
                model.train()
                optimizer.zero_grad()
                output = model(ts_embedding)
                # recon, mu, logvar = model(ts_embedding)
                # recon_loss = criterion(recon, ts_embedding)
                # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # beta = 1.0   # 可调
                # loss = recon_loss + beta * kl_loss
                loss = criterion(output, ts_embedding)
                loss.backward()
                optimizer.step()
                
                # 保存最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model_state = model.state_dict().copy()
                
                pbar.set_description(f'Epoch {epoch}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}')
            
            # 加载最佳模型参数
            model.load_state_dict(best_model_state)
            joblib.dump(model, model_cache)
        else:
            model = joblib.load(model_cache)
        model.eval()
        with torch.no_grad():
            low_dim_features = model.get_encoding(ts_embedding)  # shape (20, encoding_dim)

        csv_cache = model_cache.replace(".joblib", ".csv")
        features_np = low_dim_features.numpy()  # shape: (N, 20)
        # 转置后 shape = (20, N)，每行对应一个氨基酸
        features_np_T = features_np.T
        # 列名
        column_names = [f"Sample_{i+1}" for i in range(features_np_T.shape[1])]
        # 行索引对应氨基酸
        row_names = list("ACDEFGHIKLMNPQRSTVWY")
        # 转 DataFrame
        df = pd.DataFrame(features_np_T, columns=row_names)
        # 保存 CSV
        df.to_csv(csv_cache, index=False)


        return low_dim_features.numpy()

    def encode_onehot(self):
        onehot_embedding = np.eye(20, dtype=np.float32)
        return onehot_embedding


    def get_seq_features_np(self, at_seq, sr_seq):
        ret = np.zeros((self.ae_dim, 2, len(at_seq)), dtype=np.float32)
        for i, (a1, a2) in enumerate(zip(at_seq, sr_seq)):
            if a1 in self.AA_LIST:
                ret[:, 0, i] = self.embedding[self.AA_LIST.index(a1)]
            else:
                # TODO 这里要改成0吗
                # ret[:, 0, i] = np.nan
                ret[:, 0, i] = 0
            if a2 in self.AA_LIST:
                ret[:, 1, i] = self.embedding[self.AA_LIST.index(a2)]
            else:
                # ret[:, 1, i] = np.nan
                ret[:, 1, i] = 0
        return ret


    def construct_feature(self, at_seqs, sr_seqs):
        # feature = Parallel(n_jobs=self.n_jobs)(delayed(self.get_seq_features_np)(at_seq, sr_seq)
        #     for at_seq, sr_seq in tqdm.tqdm(zip(at_seqs, sr_seqs), total=len(at_seqs)))
        feature = [
            self.get_seq_features_np(at_seq, sr_seq)
            for at_seq, sr_seq in tqdm.tqdm(zip(at_seqs, sr_seqs), total=len(at_seqs))
        ]
        feature = np.array(feature)
        return feature
