import sys,os
import pandas as pd
import pickle
import argparse

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)


base_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V13"
eval_pickle_file = "EVAL_sn_v13_occultation_small_on_occultation_small_epoch_50_eval.dat"
proportions_file = "tmp/proportions.csv"
item_split_pickle_file = "tmp/item_split.dat"

def get_evals(eval):

    cd = {}
    for model,values in eval.items():
        cd.setdefault(model, {})
        for viewport, distances in values.items():
            cd[model][viewport] = distances['cd']

    emd = {}
    for model,values in eval.items():
        emd.setdefault(model, {})
        for viewport, distances in values.items():
            emd[model][viewport] = distances['emd']

    f_score = {}
    for model,values in eval.items():
        f_score.setdefault(model, {})
        for viewport, distances in values.items():
            f_score[model][viewport] = distances['f_score']

    return cd, emd, f_score



eval = pickle.load(open(eval_pickle_file, 'rb'))
cd, emd, f_score = get_evals(eval)

per_viewport_details = []

for key, values in emd.items():
    for viewport, value in values.items():
        details = {"file": base_path + "/" + key + "/rendering/" + viewport + ".png", "emd": value}
        per_viewport_details.append(details)

df = pd.DataFrame(per_viewport_details)

df_proportions = pd.read_csv('proportions.csv')
item_split = pickle.load(open(item_split_pickle_file, 'rb'))
df_split = pd.DataFrame.from_dict(item_split,orient='index', columns=['middle_split'])

#Merge three df informations together
df = pd.merge(df, df_proportions, how='left', left_on="file", right_on="Image file")
df = pd.merge(df, df_split, how='left', left_on="file", right_index=True)

df["oip after masking"] = df["oip after masking"] / 100
df["object to image proportion (oip)"] = df["object to image proportion (oip)"] / 100
df['crop'] = 1 - abs( df["oip after masking"]/df["object to image proportion (oip)"])
df = df.drop(['Image file', 'oip after masking', 'object to image proportion (oip)' ], axis=1)
df = df.sort_values(by=['file'])

# df = pd.merge(df, df_proportions, how='left', left_on="file", right_on="Image file")
# df = pd.merge(df, df_split, how='left', left_on="file", right_index=True)
# df = df.drop(['oip after masking', 'object to image proportion (oip)', 'mean percentage cutout' ], axis=1)
# df = df.rename(columns={"percentage cutout": "crop"})

middle_split_loss, verge_split_loss = df.groupby('middle_split')['emd'].mean()
middle_split_crop, verge_split_crop = df.groupby('middle_split')['crop'].mean()

df['full'] = [
    '#items: {}'.format(df["emd"].count()),
    'mean crop: {}'.format(df['crop'].mean()),
    'mean loss: {}'.format(df["emd"].mean())
    ]+['']*(len(df)-3)

df['middle split'] = [
    '#items: {}'.format((df['middle_split']).sum()),
    'mean crop: {}'.format(middle_split_crop),
    'mean loss: {}'.format(middle_split_loss)
    ]+['']*(len(df)-3)

df['verge split'] = [
    '#items: {}'.format((~df['middle_split']).sum()),
    'mean crop: {}'.format(verge_split_crop),
    'mean loss: {}'.format(verge_split_loss)
    ]+['']*(len(df)-3)

between_00_10 = df[(0 <= df['crop']) & (df['crop'] < 0.1)]
between_10_20 = df[(0.1 <= df['crop']) & (df['crop'] < 0.2)]
between_20_30 = df[(0.2 <= df['crop']) & (df['crop'] < 0.3)]
between_30_40 = df[(0.3 <= df['crop']) & (df['crop'] < 0.4)]
between_40_50 = df[(0.4 <= df['crop']) & (df['crop'] < 0.5)]
between_50_xx = df[(0.5 <= df['crop']) & (df['crop'] < 1)]


emd0_1 = between_00_10['emd'].mean()
emd1_2 = between_10_20['emd'].mean()
emd2_3 = between_20_30['emd'].mean()
emd3_4 = between_30_40['emd'].mean()
emd4_5 = between_40_50['emd'].mean()
emd5_x = between_50_xx['emd'].mean()

count0_1 = between_00_10['emd'].count()
count1_2 = between_10_20['emd'].count()
count2_3 = between_20_30['emd'].count()
count3_4 = between_30_40['emd'].count()
count4_5 = between_40_50['emd'].count()
count5_x = between_50_xx['emd'].count()

df['loss per crop percentage'] = [
    '0-10% - #items: {} - crop: {}'.format(count0_1, emd0_1),
    '10-20% - #items: {} - crop: {}'.format(count1_2, emd1_2),
    '20-30% - #items: {} - crop: {}'.format(count2_3, emd2_3),
    '30-40% - #items: {} - crop: {}'.format(count3_4, emd3_4),
    '40-50% - #items: {} - crop: {}'.format(count4_5, emd4_5),
    '50-100% - #items: {} - crop: {}'.format(count5_x, emd5_x)
    ]+['']*(len(df)-6)


cols = ['file', 'emd', 'middle_split', 'crop', 'full', 'middle split', 'verge split', 'loss per crop percentage']
df = df[cols]#.sort_values(by=['file'])

print(df.head(6))

df.to_csv("eval_occultation.csv", index=False)
df.to_excel("eval_occultation.xlsx")
