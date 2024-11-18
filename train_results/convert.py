import torch, os
import argparse
from tokenmix2.misc import get_env_conf


parser = argparse.ArgumentParser()
parser.add_argument("--env_conf", type=str, default=None)
args = parser.parse_args()

env_conf = get_env_conf(args.env_conf)
json_name = args.env_conf.split('/')[-1]
pth_name = json_name.replace("json", "pth")

results = [None for _ in range(32)]

base_dir = os.path.join("train_results", json_name)
for file in os.listdir(base_dir):
    layer_idx = int(file.split('.')[0])
    results[layer_idx] = torch.load(os.path.join(base_dir, file))

results[0] = None
results[1] = None

results2 = []

for x in results:
    if x is not None:
        if isinstance(x, (list, tuple)):
            results2 += x
        else:
            results2.append(x)


if not os.path.exists('ckp'):
    os.mkdir('ckp')

torch.save(results2, os.path.join("ckp", pth_name))
print(f"done!")
