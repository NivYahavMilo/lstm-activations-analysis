import os
import config
import matplotlib.pyplot as plt

import pandas as pd



def plot_rest_clip_correlation_bar():
    res = {}
    res_path = config.RESULTS_PATH_NETWORKS
    for net in os.listdir(res_path):
        rest_clip_corr = pd.read_csv(os.path.join(res_path, net))
        net_name = net.split()[1].replace('corr','')
        score = round(rest_clip_corr.loc[0].at['rest'],2)
        res[net_name] = score


    plt.bar(*zip(*res.items()))
    plt.title("Movies Correlation Between clip-rest Networks")
    plt.ylabel("Score")
    plt.xlabel("Networks")
    plt.xticks(rotation=60)
    plt.show()




if __name__ == '__main__':

    plot_rest_clip_correlation_bar()