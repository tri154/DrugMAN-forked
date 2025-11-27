import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if __name__ == "__main__":
    target_emb = pd.read_csv("target_features.tsv", index_col=0, delimiter="\t")
    X = target_emb.to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=768)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, index=target_emb.index)
    df_pca.to_csv("target_features_pca.tsv", sep="\t")
