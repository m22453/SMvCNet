# -*- coding: utf-8 -*-
# @Time : 2021/12/15 16:37
# @Author : ruinabai_TEXTCCI
# @FileName: tsne.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np

root_path = '/remote-home/TCCI17/code/RWDCL/results/CUB_tau_0.5_bs_1024_lr_0.0005/visualize/'
# paths = ['embeddings_ours_data_0.npy','embeddings_ours_0.npy', 'embeddings_ours_199.npy', 'embeddings_divide_199.npy']
paths = ['embeddings_ours_view0_0.npy', 'embeddings_ours_view0_199.npy', 'embeddings_ours_view1_0.npy','embeddings_ours_view1_199.npy']
# titles = ['original data structure', 'initialized data structure', 'data structure via RWCL', 'data structure via DIVIDE']
titles = ['initialized 1st view structure ', 'final 1st view structure','initialized 2nd view structure ', 'final 2nd view structure']
# titles = ['Fused view of BBC', 'Original view 1 of BBC','Original view 2 of BBC','Trained view 1 of BBC','Trained view 2 of BBC']

for i in range(len(paths)):
    data = np.load(root_path+paths[i])
    target = np.load(root_path + 'embeddings_divide_label.npy')
    target1 = np.load(root_path + 'embeddings_ours_label.npy')
    assert (target == target1).all()
    X_tsne = TSNE(n_components=2, random_state=100).fit_transform(data)
    print(set(target))
    # X_pca = PCA(n_components=2).fit_transform(data)

    # ckpt_dir = "images"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    #
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target, label="t-SNE")
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target, label="PCA")
    # plt.legend()
    # # plt.savefig('images/digits_tsne-pca.png', dpi=120)
    # plt.show()

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    title_str = titles[i]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    hue=target,
                    # style=target,
                    palette="tab10",
                    legend=True
                    )

    plt.title(title_str)
    plt.tight_layout()  # 调整整体空白
    plt.savefig(root_path + title_str + ".png", format='png', dpi=150)
    # plt.show()

