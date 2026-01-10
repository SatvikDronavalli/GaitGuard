import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#-------------GRF--------------
GRF_right_path = Path(__file__).parent / "GRF_F_V_PRO_right.csv"
GRF_left_path = Path(__file__).parent / "GRF_F_V_PRO_left.csv"
metadata_path = Path(__file__).parent / "GRF_metadata.csv"
GRF_right = pd.read_csv(GRF_right_path)
GRF_left = pd.read_csv(GRF_left_path)
metadata = pd.read_csv(metadata_path)
# average data if necessary instead of dropping duplicates
metadata = metadata.drop_duplicates(subset=["SUBJECT_ID"])
metadata = metadata[metadata["CLASS_LABEL"].isin(["HC","H"])]
metadata["at risk of falls"] = [None] * len(metadata)
metadata["at risk of falls"] = metadata["CLASS_LABEL"].apply(lambda x: 0 if x == "HC" else 1)
metadata["SUBJECT_ID"] = metadata["SUBJECT_ID"].astype(np.int64)
valid_IDs = metadata["SUBJECT_ID"].tolist()
# Deleting extra rows
GRF_right = GRF_right[GRF_right["SUBJECT_ID"].isin(valid_IDs)]
GRF_left = GRF_left[GRF_left["SUBJECT_ID"].isin(valid_IDs)]
# Merging 'at risk of falls' information to force data
GRF_right = GRF_right.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST","at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_left = GRF_left.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST", "at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_right_train = GRF_right[GRF_right["TRAIN_BALANCED"] == 1]
GRF_right_test = GRF_right[GRF_right["TEST"] == 1]
GRF_left_train = GRF_left[GRF_left["TRAIN_BALANCED"] == 1]
GRF_left_test = GRF_left[GRF_left["TEST"] == 1]
GRF_right_train = GRF_right_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_right_test = GRF_right_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_train = GRF_left_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_test = GRF_left_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
gait_x_train = np.stack([GRF_right_train.drop(columns="at risk of falls"),GRF_left_train.drop(columns="at risk of falls")],-1)
gait_y_train = GRF_right_train["at risk of falls"]
gait_x_test = np.stack([GRF_right_test.drop(columns="at risk of falls"),GRF_left_test.drop(columns="at risk of falls")],-1)
gait_y_test = GRF_right_test["at risk of falls"]
#-------------ML-COP--------------
GF_right_path = Path(__file__).parent / "GRF_COP_ML_PRO_right.csv"
GRF_left_path = Path(__file__).parent / "GRF_COP_ML_PRO_left.csv"
metadata_path = Path(__file__).parent / "GRF_metadata.csv"
GRF_right = pd.read_csv(GRF_right_path)
GRF_left = pd.read_csv(GRF_left_path)
metadata = pd.read_csv(metadata_path)
# average data if necessary instead of dropping duplicates
metadata = metadata.drop_duplicates(subset=["SUBJECT_ID"])
metadata = metadata[metadata["CLASS_LABEL"].isin(["HC","H"])]
metadata["at risk of falls"] = [None] * len(metadata)
metadata["at risk of falls"] = metadata["CLASS_LABEL"].apply(lambda x: 0 if x == "HC" else 1)
metadata["SUBJECT_ID"] = metadata["SUBJECT_ID"].astype(np.int64)
valid_IDs = metadata["SUBJECT_ID"].tolist()
# Deleting extra rows
GRF_right = GRF_right[GRF_right["SUBJECT_ID"].isin(valid_IDs)]
GRF_left = GRF_left[GRF_left["SUBJECT_ID"].isin(valid_IDs)]
# Merging 'at risk of falls' information to force data
GRF_right = GRF_right.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST","at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_left = GRF_left.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST", "at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_right_train = GRF_right[GRF_right["TRAIN_BALANCED"] == 1]
GRF_right_test = GRF_right[GRF_right["TEST"] == 1]
GRF_left_train = GRF_left[GRF_left["TRAIN_BALANCED"] == 1]
GRF_left_test = GRF_left[GRF_left["TEST"] == 1]
GRF_right_train = GRF_right_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_right_test = GRF_right_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_train = GRF_left_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_test = GRF_left_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
#print(gait_x_train.shape,np.stack([GRF_right_train.drop(columns="at risk of falls"),GRF_left_train.drop(columns="at risk of falls")],-1).shape)
combined_ml_cop_train = np.stack([GRF_right_train.drop(columns="at risk of falls"),GRF_left_train.drop(columns="at risk of falls")],-1)
combined_ml_cop_test = np.stack([GRF_right_test.drop(columns="at risk of falls"),GRF_left_test.drop(columns="at risk of falls")],-1)
#-------------AP-COP--------------
GF_right_path = Path(__file__).parent / "GRF_COP_AP_PRO_right.csv"
GRF_left_path = Path(__file__).parent / "GRF_COP_AP_PRO_left.csv"
metadata_path = Path(__file__).parent / "GRF_metadata.csv"
GRF_right = pd.read_csv(GRF_right_path)
GRF_left = pd.read_csv(GRF_left_path)
metadata = pd.read_csv(metadata_path)
# average data if necessary instead of dropping duplicates
metadata = metadata.drop_duplicates(subset=["SUBJECT_ID"])
metadata = metadata[metadata["CLASS_LABEL"].isin(["HC","H"])]
metadata["at risk of falls"] = [None] * len(metadata)
metadata["at risk of falls"] = metadata["CLASS_LABEL"].apply(lambda x: 0 if x == "HC" else 1)
metadata["SUBJECT_ID"] = metadata["SUBJECT_ID"].astype(np.int64)
valid_IDs = metadata["SUBJECT_ID"].tolist()
# Deleting extra rows
GRF_right = GRF_right[GRF_right["SUBJECT_ID"].isin(valid_IDs)]
GRF_left = GRF_left[GRF_left["SUBJECT_ID"].isin(valid_IDs)]
# Merging 'at risk of falls' information to force data
GRF_right = GRF_right.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST","at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_left = GRF_left.merge(metadata[["SUBJECT_ID","TRAIN_BALANCED", "TEST", "at risk of falls"]], on="SUBJECT_ID", how="inner")
GRF_right_train = GRF_right[GRF_right["TRAIN_BALANCED"] == 1]
GRF_right_test = GRF_right[GRF_right["TEST"] == 1]
GRF_left_train = GRF_left[GRF_left["TRAIN_BALANCED"] == 1]
GRF_left_test = GRF_left[GRF_left["TEST"] == 1]
GRF_right_train = GRF_right_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_right_test = GRF_right_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_train = GRF_left_train.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
GRF_left_test = GRF_left_test.drop(columns=["TRAIN_BALANCED","TEST","SUBJECT_ID","SESSION_ID","TRIAL_ID"])
#print(gait_x_train.shape,np.stack([GRF_right_train.drop(columns="at risk of falls"),GRF_left_train.drop(columns="at risk of falls")],-1).shape)
combined_ap_cop_train = np.stack([GRF_right_train.drop(columns="at risk of falls"),GRF_left_train.drop(columns="at risk of falls")],-1)
combined_ap_cop_test = np.stack([GRF_right_test.drop(columns="at risk of falls"),GRF_left_test.drop(columns="at risk of falls")],-1)
final_x_train = np.concatenate([combined_ml_cop_train,combined_ap_cop_train,gait_x_train],-1)
final_y_train = GRF_right_train["at risk of falls"]
final_x_test = np.concatenate([combined_ml_cop_test,combined_ap_cop_test,gait_x_test],-1)
final_y_test = GRF_right_test["at risk of falls"]
print(final_x_train.shape,final_y_train.shape,final_x_test.shape,final_y_test.shape)
# All desired subjects are in the GRF dataframes according to the checks below
#print(len(metadata[metadata["TRAIN_BALANCED"] == 1]), len(metadata[metadata["TRAIN_BALANCED"] == 0]))
#print(set(GRF_left["SUBJECT_ID"]) == set(metadata["SUBJECT_ID"]))
#print(set(GRF_right["SUBJECT_ID"]) == set(metadata["SUBJECT_ID"]))