import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("../New_Data/HC101_SelfPace.csv")
df = df[df["GeneralEvent"] != 'Standing']
metadata = pd.read_csv("CONTROLS - Demographic+Clinical - datasetV1.csv")
df_copy = df[df["GeneralEvent"] == 'Walk']
plt.plot(df_copy['RTotalForce'])
idx = 0
for i in df["GeneralEvent"]:
    if i == 'Turn':
        break
    idx += 1
df = df.iloc[1:idx]
plt.plot(df["RTotalForce"])

cases = []
while True:
    stepStarted = False
    bad_curve = False
    started_idx = 0
    ended_idx = 0
    for i in range(0, len(df["RTotalForce"])):
        if not stepStarted and df.iloc[i]["RTotalForce"] < 40 and df.iloc[i+1]["RTotalForce"] >= 40:
            stepStarted = True
            started_idx = i
            ended_idx = started_idx
        elif stepStarted and df.iloc[i]["RTotalForce"] < 40 and df.iloc[i+1]["RTotalForce"] < 40:
            break
        elif stepStarted:
            ended_idx += 1
    new_df = df.iloc[started_idx:ended_idx+2]
    bad_indexes = list(new_df[new_df["RTotalForce"].isna()].index)
    if not bad_indexes:
        cases.append((started_idx,ended_idx+2))
    plt.plot(new_df["RTotalForce"])
    plt.show()
    df = df.iloc[ended_idx+2:]
    if len(df) <= 100:
        break
print("completed extraction")

#df.to_csv("test.csv")
