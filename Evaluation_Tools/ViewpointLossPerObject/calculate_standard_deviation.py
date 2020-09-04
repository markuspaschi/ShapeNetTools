import sys,os,re
import pandas as pd
import pickle
import argparse

curr_dir = os.getcwd()

#TODO: make eval dirs as arguments passable
eval_2 = pickle.load(open("/Users/markuspaschke/Desktop/presentation/_pmon3/full_evals/sn_v13_on_2.5D_epoch_50_full_eval.dat", 'rb'))
eval_2_5 = pickle.load(open("/Users/markuspaschke/Desktop/presentation/_pmon3/full_evals/2.5D_on_2.5D_epoch_50_full_eval.dat", 'rb'))


def save_to_csv_xlsx(eval_2, eval_2_5, eval_dir, ignore_styling):
    def clamp(x):
      return max(0, min(x, 255))

    def color_calculation(value, mean, color_factor):

        pattern = '=HYPERLINK\(.* & ".*"[,;] \"(nan|[-+]?[0-9]*\.?[0-9]+)\"\)'
        match = re.match(pattern, value)

        if(match is None):
            value = float("nan")
        else:
            value = float(match.group(1))

        if(value == 0):
            #perfect (ground_truth)
            return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(255), clamp(0))

        ratio = min(abs(value - mean) / mean / color_factor, 1)
        r = 2 * int(max(0, 255*ratio))
        g = 2 * int(max(0, 255*(1 - ratio)))
        b = 0
        return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

    def std_color(ratio):
        r,g,b = 0,255,0
        return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

    def blue_color():
        r,g,b = 102,78,255
        return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

    def white_color():
        r,g,b = 255,255,255
        return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

    def cv_color(ratio, color_factor):
        ratio = min(ratio / (color_factor), 1)
        r = 2 * int(max(0, 255*ratio))
        g = 2 * int(max(0, 255*(1 - ratio)))
        b = 0
        # b = int(max(0, 255*(ratio - 1)))
        # g = int(max(0, 255*(1 - ratio)))
        # r = 255 - b - g
        return "background-color: #{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

    def color_to_mean(val, color_factor):

        if("training_loss" in val.loc['type']):
            color_factor * 0.5

        mean = val.loc['mean']

        # 0 -> file
        # next 1:-7 are viewports
        # type, mean, std, cv, PATH,PATH,PATH

        # TODO dynamic (with regex) for column names
        colors = [white_color()] + [color_calculation(x, mean, color_factor) for x in val[1:-7]] + [blue_color()] + [std_color(val[-6])] + [std_color(val[-5])] + [cv_color(val[-4], color_factor)] + [x] * 3
        return colors

    def make_hyperlink(file, viewport, value, path_cell, is_xyz=False):

        if(is_xyz):
            return '=HYPERLINK(%s & "/%s/rendering/%s.xyz", "%s")' % (path_cell,file,viewport,value)
        else:
            return '=HYPERLINK(%s & "/%s/rendering/%s.obj", "%s")' % (path_cell,file,viewport,value)

    def link_to_files(df, path_cell, is_xyz=False):
        for idx, row in df.iterrows():
            for column_idx, value in enumerate(row):
                column_name = df.columns[column_idx]
                if re.match("^\d+", column_name):
                    df.loc[idx, column_name] = make_hyperlink(row.name, column_name, value, path_cell, is_xyz)
        return df

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

        training_loss = {}
        for model,values in eval.items():
            training_loss.setdefault(model, {})
            for viewport, distances in values.items():
                training_loss[model][viewport] = distances['full_loss']

        return cd, emd, f_score, training_loss

    def into_df(dict, type):
        df = pd.DataFrame.from_dict(dict).T
        df = df.reindex(sorted(df.columns), axis=1)
        df['type'] = type
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)
        df['coefficient of variation (cv)'] = df['std'] / df['mean']
        return df

    ## CREATE CD EMD EVAL
    cd_2, emd_2, f_score_2, training_loss_2 = get_evals(eval_2)
    cd2_5, emd2_5, f_score2_5, training_loss2_5 = get_evals(eval_2_5)

    df1 = into_df(emd_2, "2D - emd")
    df2 = into_df(training_loss_2, "2D - training_loss")
    df3 = into_df(emd2_5, "2_5D - emd")
    df4 = into_df(training_loss2_5, "2_5D - training_loss")

    # TODO: Make this dynamic and not hardcoded
    df1 = link_to_files(df1, "AA2")
    df2 = link_to_files(df2, "AA2")
    df3 = link_to_files(df3, "AB2")
    df4 = link_to_files(df4, "AB2")

    # GROUND_TRUTH
    df_gt = df1.copy()
    df_gt['type'] = "GROUND_TRUTH"
    for col in df_gt.filter(regex=("(\d+|coefficient of variation \(cv\)|mean|std)")).columns:
        df_gt[col].values[:] = 0.0
    df_gt = link_to_files(df_gt, "AC2", is_xyz=True)

    full_df = pd.concat([df1,df2,df3,df4,df_gt])
    full_df.index.name = 'file'
    full_df = full_df.sort_values(by=['file', 'type'])

    full_df['PATH_2D should start with file://'] = ["FILL OUT THIS PATH"] + (len(full_df) -1) * [""]
    full_df['PATH_2_5D should start with file://'] = ["FILL OUT THIS PATH"] + (len(full_df) -1) * [""]
    full_df['PATH_GROUND_TRUTH should start with file://'] = ["FILL OUT THIS PATH"] + (len(full_df) -1) * [""]

    full_df = full_df.reset_index()

    df_colored = full_df.style.apply(lambda x : color_to_mean(x, color_factor = 0.8), axis=1)
    df_colored.to_excel(os.path.join(eval_dir, 'standard_deviation_full.xlsx'))

    print("Finished")


def get_args():
    parser = argparse.ArgumentParser(description='Enter additional args')
    parser.add_argument('--eval_name', type=str, required=True, help='Name for evaluation')
    parser.add_argument('--ignore_styling', type=bool, required=False, default=False, help='Requires Jinja2 and openpyxl and maybe some other dependencies')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(curr_dir, args.eval_name)):
        sys.exit("Eval name does not exist!")

    return args

args = get_args()
eval_dir = os.path.join(curr_dir, args.eval_name)

save_to_csv_xlsx(eval_2, eval_2_5, eval_dir, args.ignore_styling)
