from tokenmix2.misc import get_model_and_tokenizer
from tokenmix2.misc import get_env_conf
from tokenmix2.misc import Evaluator
import argparse, os

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--use_env_conf_tasks", action="store_true", default=False)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    test_conf = get_env_conf("test_draft/eval.json")

    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    model.eval()
    model.model.always_return_attn_score()


    def callback(outputs):
        N_CLUSTER = 64
        N_DATA = 32

        # extract attentions
        attn = torch.cat(outputs.attentions, dim=0)[..., :N_DATA, :N_DATA][2:, ...]
        attn = attn.flatten(0,1).flatten(-2,-1).float().numpy()

        # use tsne to cluster the data points
        tsne = TSNE(n_components=2, random_state=0)
        attn_tsne = tsne.fit_transform(attn).reshape(30, -1, 2)
        
        # attn_tsne: [32, 32, 4096]
        attn_tsne_1 = attn_tsne[:2, ...].reshape(-1, 2)
        attn_tsne_2 = attn_tsne[2:, ...].reshape(-1, 2)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.scatter(attn_tsne_1[:, 0], attn_tsne_1[:, 1], c='red', marker='X', s=100, label='Cluster centers')
        plt.scatter(attn_tsne_2[:, 0], attn_tsne_2[:, 1], alpha=0.6, label="Data points")
        plt.savefig("scatter.jpg", dpi=640)

        import IPython
        IPython.embed(header='debug')


    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    evaluator_class = Evaluator

    if args.use_env_conf_tasks:
        evaluator = evaluator_class(model, tokenizer, **env_conf["train"], callback=callback)
    else:
        evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf, callback=callback)
    
    evaluator.evaluate()
