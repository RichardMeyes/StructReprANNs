import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import deepdish as dd

from train_test_net import Net


def plot_tSNE(testloader, labels, num_samples, name=None, title=None):
    X_img = testloader.dataset.test_data.numpy()[:num_samples]
    X_img_label = testloader.dataset.test_labels.numpy()[:num_samples]

    print("loading fitted tSNE coordinates...")
    X_tsne = pickle.load(open("../data/tSNE/X_tSNE_10000.p".format(num_samples), "rb"))

    print("plotting tSNE...")
    # scaling
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    t0 = time.time()

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    ax = fig.add_subplot(111)

    # Define custom color maps
    custom_cmap_black = plt.cm.Greys
    custom_cmap_black_colors = custom_cmap_black(np.arange(custom_cmap_black.N))
    custom_cmap_black_colors[:, -1] = np.linspace(0, 1, custom_cmap_black.N)
    custom_cmap_black = ListedColormap(custom_cmap_black_colors)

    custom_cmap_red = plt.cm.bwr
    custom_cmap_red_colors = custom_cmap_red(np.arange(custom_cmap_red.N))
    custom_cmap_red_colors[:, -1] = np.linspace(0, 1, custom_cmap_red.N)
    custom_cmap_red = ListedColormap(custom_cmap_red_colors)

    custom_cmap_white = plt.cm.Greys
    custom_cmap_white_colors = custom_cmap_white(np.arange(custom_cmap_white.N))
    custom_cmap_white_colors[:, -1] = 0
    custom_cmap_white = ListedColormap(custom_cmap_white_colors)

    custom_cmap_green = plt.cm.brg
    custom_cmap_green_colors = custom_cmap_green(np.arange(custom_cmap_green.N))
    custom_cmap_green_colors[:, -1] = np.linspace(0, 1, custom_cmap_green.N)
    custom_cmap_green = ListedColormap(custom_cmap_green_colors)

    color_maps = [custom_cmap_red, custom_cmap_black, custom_cmap_green, custom_cmap_white]

    if hasattr(offsetbox, "AnnotationBbox"):
        for i_digit in range(num_samples):
            # correct color for plotting
            X_img[i_digit][X_img[i_digit, :, :] > 10] = 255
            X_img[i_digit][X_img[i_digit, :, :] <= 10] = 0
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X_img[i_digit],
                                                                      cmap=color_maps[labels[i_digit]],
                                                                      zoom=0.25),
                                                X_tsne[i_digit],
                                                frameon=False,
                                                pad=0)
            ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
    # save figure
    plt.savefig("../plots/MNIST_tSNE_{0}_{1}.pdf".format(num_samples, name), dpi=300)
    t1 = time.time()
    print("done! {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":

    # setting flags
    flag_plot_tSNE = False

    # setting rng seed for reproducibility
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load nets and weights
    layers = (100, 100, 100)
    net_trained = Net(layers)
    net_untrained = Net(layers)
    # send nets to GPU
    for net in [net_trained, net_untrained]:
        net.to(device)
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    net_trained.load_state_dict(torch.load("../nets/MNIST_MLP_{0}_trained.pt".format(layers)))
    net_trained.eval()
    net_untrained.load_state_dict(torch.load("../nets/MNIST_MLP_{0}_untrained.pt".format(layers)))
    net_untrained.eval()

    # load data and test network
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    acc_full, labels, acc_class_full, _ = net_trained.test_net(criterion, testloader, device)
    print(acc_full)
    print(labels)
    print(acc_class_full)
    print(acc_class_full.mean())
    if flag_plot_tSNE:
        plot_tSNE(testloader, labels, num_samples=10000, name="", title="accuray: {0}%".format(acc_full))

    ko_layers = np.arange(len(layers))
    ko_results = dict()
    for ko_layer in ko_layers:
        ko_results["layer_{0}".format(ko_layer)] = dict()
        ko_units = np.arange(layers[ko_layer])
        for ko_unit in ko_units:
            print("knockout layer {0}, unit {1}".format(ko_layer, ko_unit))
            net_trained.load_state_dict(torch.load("../nets/MNIST_MLP_{0}_trained.pt".format(layers)))
            net_trained.eval()

            n_inputs = layers[ko_layer-1] if ko_layer != 0 else 784
            net_trained.__getattr__("h{0}".format(ko_layer)).weight.data[ko_unit, :] = torch.zeros(n_inputs)
            net_trained.__getattr__("h{0}".format(ko_layer)).bias.data[ko_unit] = 0
            acc, labels, acc_class, labels_class = net_trained.test_net(criterion, testloader, device)

            # append to results dict
            ko_results["layer_{0}".format(ko_layer)]["unit_{0}".format(ko_unit)] = {"acc": np.array([acc]),
                                                                                    "labels": labels,
                                                                                    "acc_class": acc_class,
                                                                                    "labels_class": labels_class}

    dd.io.save("../data/results/ko_results.h5", ko_results, compression=None)
