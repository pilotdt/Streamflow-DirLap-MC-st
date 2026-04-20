python3 src/main_dcrnn.py  reg_4_loss=L_dir add_storage=True horizon=7 history=7
python3 src/main_mtgnn.py  reg_4_loss=L_dir add_storage=True horizon=7 history=7

python3 src/main_dcrnn.py adj_np="data/undirected-adj-mat-all-stations.csv"   binary_adj=True  un_adj=True reg_4_loss=L_dir add_storage=True horizon=7 history=7
python3 src/main_mtgnn.py adj_np="data/undirected-adj-mat-all-stations.csv"   binary_adj=True  un_adj=True reg_4_loss=L_dir add_storage=True horizon=7 history=7

python3 src/main_dcrnn.py adj_np="data/binary-adj-mat-all-stations.csv"   binary_adj=True  reg_4_loss=L_dir add_storage=True horizon=7 history=7
python3 src/main_mtgnn.py adj_np="data/binary-adj-mat-all-stations.csv"   binary_adj=True  reg_4_loss=L_dir add_storage=True horizon=7 history=7
