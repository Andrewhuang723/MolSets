import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt



def start_plot(figsize=(10, 8), style = 'whitegrid', dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1,1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0,0])
    return ax

def R2_plot(target_name, predict_name, df, prop_name_T, ax=None):
    if ax is None:
        ax = start_plot(style='darkgrid', dpi=180)
    reconstruct_y_test = df[target_name].values.reshape(-1)
    reconstruct_y_pred = df[predict_name].values.reshape(-1)
    error = df["error"].mean()
    accuracy = len(df[df["error"] < 0.1]) / len(df)
    sns.set(rc={'font.family': 'Times New Roman'})
    good_acc_df = df.loc[df["error"] <= 0.1]
    bad_acc_df = df.loc[df["error"] > 0.1]
    ax.scatter(good_acc_df[target_name], good_acc_df[predict_name], color='darkorange', edgecolor='navy',
               label=f"Good Accuracy: {len(good_acc_df)}")
    ax.scatter(bad_acc_df[target_name], bad_acc_df[predict_name], marker='x', edgecolor='navy',
               label=f"Bad Accuracy: {len(bad_acc_df)}")
    ymin = min(np.min(reconstruct_y_test), np.min(reconstruct_y_pred)) - 0.1
    ymax = max(np.max(reconstruct_y_test), np.max(reconstruct_y_pred)) + 0.1
    lim = [ymin, ymax]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.plot(lim, lim, c='brown', ls='--', label=r'$y=\hat y, $' + 'identity')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=15,
              title=r'$R^2$' + ": %.4f" % r2_score(y_true=reconstruct_y_test, y_pred=reconstruct_y_pred) + '\n' +
                     "MAPE" + ": %.4f" % error + '\n' +
                     "Accuracy" + ": %.4f" % accuracy,
                title_fontsize=15)
    plt.xlabel('TRUE %s' % prop_name_T, fontsize=20, font="Times New Roman")
    plt.ylabel('PREDICTED %s' % prop_name_T, fontsize=20, font="Times New Roman")
    plt.xticks(fontsize=20, font="Times New Roman")
    plt.yticks(fontsize=20, font="Times New Roman")
    plt.title('%s Testing: %d' % (prop_name_T, len(reconstruct_y_test)), fontsize=20, font="Times New Roman")
    return ax