from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from scatwave.scattering import Scattering

DATASETS = ["mnist", "fashion-mnist", "cifar-10"]

def get_scatter_features(data_loader, dataset_name, num_layers=2):
    scatter_features = []
    targets = []

    avg_pool = nn.AvgPool3d(2, stride=2) # for cifar-10

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        # Scattering params
        # M, N: input image size
        # J: number of layers
        data = data.cuda()
        # print(data.size())
        # print(data.size(data.dim()-2), data.size(data.dim()-1))
        scat = Scattering(M=data.size(data.dim()-2), N=data.size(data.dim()-1), J=num_layers).cuda()
        out = scat(data)
        out = out.squeeze()
        # print("out", out.size())

        for i in range(len(out)):
            feat = out[i]
            scatter_features.append(feat.cpu().numpy())
        for i in range(len(target)):
            targets.append(target[i].cpu().numpy())

    scatter_features = np.array(scatter_features)
    targets = np.array(targets)

    print("raw scatter features dim:", scatter_features.shape)

    if dataset_name == "cifar-10":
        print("cifar-10 avg pooling")
        scatter_features = avg_pool(torch.cuda.FloatTensor(scatter_features)).cpu().numpy()
        print("after pooling:", scatter_features.shape)

    total_sample = scatter_features.shape[0]
    scatter_features = scatter_features.reshape(total_sample, -1)

    print("scatter features dim:", scatter_features.shape, "target features dim:", targets.shape)
    return scatter_features, targets

def get_dataset(dataset_name="mnist", is_train=True, batch_size=128):
    transform = transforms.Compose([   
            # transforms.Scale(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name == "mnist":
        return  torch.utils.data.DataLoader(datasets.MNIST('./data/' + dataset_name, train=is_train, download=True,
                    transform=transform), batch_size=batch_size, shuffle=False)
    elif dataset_name == "fashion-mnist":
        return  torch.utils.data.DataLoader(datasets.FashionMNIST('./data/' + dataset_name, train=is_train, download=True,
                    transform=transform), batch_size=batch_size, shuffle=False)
    elif dataset_name == "cifar-10":
        return  torch.utils.data.DataLoader(datasets.CIFAR10('./data/' + dataset_name, train=is_train, download=True,
                    transform=transform), batch_size=batch_size, shuffle=False)
    else:
        return None

def generate_scatter_features(batch_size=128, n_layer=2):
    for dataset_name in DATASETS:
        train_dataset = get_dataset(dataset_name, is_train=True, batch_size=batch_size)

        print("[SCATTER] Feature extraction for", dataset_name)
        train_scatter_feats, train_targets = get_scatter_features(train_dataset, dataset_name, n_layer)

        print("save training data")
        np.save('features/train_X_scatternet_' + dataset_name + "_bsz" + str(batch_size) + "_layer" + str(n_layer) + ".npy", train_scatter_feats)
        np.save('features/train_Y_scatternet_' + dataset_name + "_bsz" + str(batch_size) + "_layer" + str(n_layer) + ".npy", train_targets)
        
        test_dataset = get_dataset(dataset_name, is_train=False, batch_size=batch_size)
        test_scatter_feats, test_targets = get_scatter_features(test_dataset, dataset_name, n_layer)

        print("save test data")
        np.save('features/test_X_scatternet_' + dataset_name + "_bsz" + str(batch_size) + "_layer" + str(n_layer) + ".npy", test_scatter_feats)
        np.save('features/test_Y_scatternet_' + dataset_name + "_bsz" + str(batch_size) + "_layer" + str(n_layer) + ".npy", test_targets)

for layer in [1,2]:
    print("-"*50)
    print("running Scatternet with layer:",layer)
    print("-"*50)
    generate_scatter_features(128, layer)
