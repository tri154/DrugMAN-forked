import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch
from sklearn.preprocessing import StandardScaler


class DrugMANDataset:
    def __init__(self, file_csv, file_emb):
        self.file_csv = file_csv
        self.file_emb = file_emb
        self.all_binds = "data/all_bind.csv"

    def load_data(self):
        part = 1
        train = pd.read_csv(self.file_csv + f"/bind_train_part{part}.csv")
        val = pd.read_csv(self.file_csv + f"/bind_val_part{part}.csv")
        test = pd.read_csv(self.file_csv + f"/bind_test_part{part}.csv")

        return train, val, test

    def load_embed(self):
        drug_emb = pd.read_csv(self.file_emb + "/drug_features.tsv", index_col=0, delimiter="\t")
        target_emb = pd.read_csv(self.file_emb + "/target_features.tsv", index_col=0, delimiter="\t")

        return drug_emb, target_emb

    def get_dataloader(self):
        # all_binds = pd.read_csv(self.all_binds)
        # all_binds = all_binds[["pubchem_cid", "gene_id"]]

        train, val, test = self.load_data()
        # train = train[train["label"] == 1]
        # val = val[val["label"] == 1]
        # test = test[test["label"] == 1]

        # all_bind_temp = pd.concat([train, val, test], axis=0)
        # all_bind_temp = all_bind_temp[["pubchem_cid", "gene_id"]]


        # all_binds = all_binds.sort_values(by=all_binds.columns.tolist()).reset_index(drop=True)
        # all_bind_temp= all_bind_temp.sort_values(by=all_bind_temp.columns.tolist()).reset_index(drop=True)

        # print(all_binds)
        # print(all_bind_temp)
        # print(all_binds.equals(all_bind_temp))

        # input()
        drug_emb, target_emb = self.load_embed()

        train_drug_emb = drug_emb.loc[train['pubchem_cid'], ]
        train_target_emb = target_emb.loc[train['gene_id'], ]

        val_drug_emb = drug_emb.loc[val["pubchem_cid"], ]
        val_target_emb = target_emb.loc[val['gene_id'], ]

        test_drug_emb = drug_emb.loc[test["pubchem_cid"], ]
        test_target_emb = target_emb.loc[test['gene_id'], ]

        # normalized by z-score and convert to tensor type
        scaler = StandardScaler()
        train_drug_emb = scaler.fit_transform(np.array(train_drug_emb))
        train_target_emb = scaler.fit_transform(np.array(train_target_emb))
        train_drug_emb = torch.FloatTensor(train_drug_emb)
        train_target_emb = torch.FloatTensor(train_target_emb)
        train_label = torch.FloatTensor(np.array(train['label']))

        val_drug_emb = scaler.fit_transform(np.array(val_drug_emb))
        val_target_emb = scaler.fit_transform(np.array(val_target_emb))
        val_drug_emb = torch.FloatTensor(val_drug_emb)
        val_target_emb = torch.FloatTensor(val_target_emb)
        val_label = torch.FloatTensor(np.array(val['label']))

        test_drug_emb = scaler.fit_transform(np.array(test_drug_emb))
        test_target_emb = scaler.fit_transform(np.array(test_target_emb))
        test_drug_emb = torch.FloatTensor(test_drug_emb)
        test_target_emb = torch.FloatTensor(test_target_emb)
        test_label = torch.FloatTensor(np.array(test['label']))

        # create dataloader
        train_dataset = Data.TensorDataset(train_drug_emb, train_target_emb, train_label)
        val_dataset = Data.TensorDataset(val_drug_emb, val_target_emb, val_label)
        test_dataset = Data.TensorDataset(test_drug_emb, test_target_emb, test_label)

        params = {'batch_size': 512, 'shuffle': True, 'num_workers': 0, 'drop_last': True}
        if train_dataset or val_dataset:
            train_loader = Data.DataLoader(train_dataset, **params)
            val_loader = Data.DataLoader(val_dataset, **params)
        if test_dataset:
            params['shuffle'] = False
            params['drop_last'] = False
            params['batch_size'] = len(test_dataset)
            test_loader = Data.DataLoader(test_dataset, **params)
            test_bcs = len(test_dataset)

        return train_loader, val_loader, test_loader, test_bcs
