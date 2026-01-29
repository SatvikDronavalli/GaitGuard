from IMU_Pods.TUG_data_visualization import dataset_preprocessing, butter_lowpass
from IMU_Pods.AI_Pipeline import normalize, resample
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib

REPO_PATH = 'C:/Users/Satvik/PycharmProjects/locked_in/IMU_Pods/'
DATASET_PATH = "dataset.csv"
'''
Index(['Patient', 'Trial_Name', 'Current_VGA', 'Unstable_Gait',
       'Trial_Number', 'Condition', 'Gait_Start', 'Gait_End', 'UTurn_Start',
       'UTurn_End', 'Data_Path'],
      dtype='object')
'''

def data_preparation(REPO_PATH, DATASET_PATH, PATIENT_INDEX):
    df = pd.read_csv(REPO_PATH + DATASET_PATH, index_col=0)
    new_df = df[df['Unstable_Gait'] == 1].iloc[PATIENT_INDEX]
    bound1 = new_df.Gait_Start
    bound2 = new_df.UTurn_Start
    bound3 = new_df.UTurn_End
    bound4 = new_df.Gait_End
    raw_data = normalize(np.load(REPO_PATH + new_df["Data_Path"]))
    walk1 = resample(raw_data[bound1:bound2], 2000)
    turn = resample(raw_data[bound2:bound3], 2000)
    walk2 = resample(raw_data[bound3:bound4], 2000)
    output = np.concatenate((walk1,turn,walk2), axis=1)
    output = np.expand_dims(output, axis=0)
    return output

wrong_cnt = 0
right_cnt = 0

for i in range(100):
    fall_input = data_preparation(REPO_PATH, DATASET_PATH, i)

    conv = keras.models.load_model('cnn_model.keras')

    feature_extractor = keras.Model(
            inputs=conv.input,
            outputs=conv.layers[-2].output
        )

    features = feature_extractor.predict(fall_input, verbose=0)
    svm_predictor = joblib.load('svm_classifier.joblib')
    predictions = svm_predictor.predict(features)
    if predictions[0] == 0:
        wrong_cnt += 1
    else:
        right_cnt += 1

print(wrong_cnt, right_cnt)