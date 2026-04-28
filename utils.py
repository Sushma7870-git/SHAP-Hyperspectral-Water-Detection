# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import io, misc
import imageio
import os
import re
import torch
import shap
import pandas as pd
from torch.utils.data import DataLoader


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def plot_image_with_legend(pred_image, class_map):
    # Define custom colors: Blue for water, Black for undefined, Brown for other classes
    custom_colors = ['black', 'blue', 'brown']  # You can add more colors as needed for more classes

    # Create a custom colormap with the defined colors
    cmap = ListedColormap(custom_colors)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12,10))

    # Display the predicted image with the custom colormap
    cax = ax.imshow(pred_image, cmap=cmap)

    # Add a colorbar with class names as tick labels
    cbar = fig.colorbar(cax, ax=ax, ticks=np.arange(len(class_map)))
    cbar.set_ticklabels(list(class_map.values()))

    # Set tick label font size to 16
    cbar.ax.tick_params(labelsize=20)

    # Add a title and disable axes for a cleaner plot
    # ax.set_title('Hamida(1D CNN)')
    ax.axis('off')

    # Save full size plot with no white space
    plt.savefig(r'C:\Users\sksus\Documents\Code\Hyperspectral\Other_Image\whu\chen.png', bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
    plt.close()  # Close figure to free memory


def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# def metrics(prediction, target, ignored_labels=[], n_classes=None):
#     """Compute and print metrics (accuracy, confusion matrix and F1 scores).

#     Args:
#         prediction: list of predicted labels
#         target: list of target labels
#         ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
#         n_classes (optional): number of classes, max(target) by default
#     Returns:
#         accuracy, F1 score by class, confusion matrix
#     """
#     ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
#     for l in ignored_labels:
#         ignored_mask[target == l] = True
#     ignored_mask = ~ignored_mask
#     #target = target[ignored_mask] -1
#     target = target[ignored_mask]
#     prediction = prediction[ignored_mask]

#     results = {}

#     n_classes = np.max(target) + 1 if n_classes is None else n_classes

#     cm = confusion_matrix(
#         target,
#         prediction,
#         labels=range(n_classes))

#     results["Confusion matrix"] = cm

#     # Compute global accuracy
#     total = np.sum(cm)
#     accuracy = sum([cm[x][x] for x in range(len(cm))])
#     accuracy *= 100 / float(total)

#     results["Accuracy"] = accuracy

#     # Compute F1 score
#     F1scores = np.zeros(len(cm))
#     for i in range(len(cm)):
#         try:
#             F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
#         except ZeroDivisionError:
#             F1 = 0.
#         F1scores[i] = F1

#     results["F1 scores"] = F1scores

#     # Compute kappa coefficient
#     pa = np.trace(cm) / float(total)
#     pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
#         float(total * total)
#     kappa = (pa - pe) / (1 - pe)
#     results["Kappa"] = kappa

#     return results

# -------------------------New Tested matrix--------------------------------------------------


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute classification metrics including Accuracy, Confusion Matrix, Precision, Recall, F1 scores, and Kappa.

    Args:
        prediction: numpy array of predicted labels
        target: numpy array of true labels
        ignored_labels (optional): list of labels to ignore (e.g., 0 for undef)
        n_classes (optional): number of classes, defaults to max(target) + 1
    Returns:
        Dictionary containing Accuracy, Precision, Recall, F1 scores, Confusion Matrix, and Kappa.
    """
    # Mask out ignored labels
    ignored_mask = ~np.isin(target, ignored_labels)
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    # Determine number of classes
    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    # Compute confusion matrix
    cm = confusion_matrix(target, prediction, labels=np.arange(n_classes))
    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm) / total * 100  # Overall accuracy in percentage
    results["Accuracy"] = accuracy

    # Initialize metric arrays
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_scores = np.zeros(n_classes)

    # Compute Precision, Recall, and F1-score per class
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP  # False Positives
        FN = np.sum(cm[i, :]) - TP  # False Negatives

        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_scores[i] = (2.0 * TP) / (2.0 * TP + FP + FN) if (2.0 * TP + FP + FN) > 0 else 0.0

    results["Precision"] = precision
    results["Recall"] = recall
    results["F1 scores"] = f1_scores

    # Compute kappa coefficient
    pa = np.trace(cm) / total
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total**2)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results



# def show_results(results, vis, label_values=None, agregated=False):
#     text = ""

#     if agregated:
#         accuracies = [r["Accuracy"] for r in results]
#         kappas = [r["Kappa"] for r in results]
#         F1_scores = [r["F1 scores"] for r in results]

#         F1_scores_mean = np.mean(F1_scores, axis=0)
#         F1_scores_std = np.std(F1_scores, axis=0)
#         cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
#         text += "Agregated results :\n"
#     else:
#         cm = results["Confusion matrix"]
#         accuracy = results["Accuracy"]
#         F1scores = results["F1 scores"]
#         kappa = results["Kappa"]

#     # Convert to percentage (row-wise normalization)
#     cm_sum = cm.sum(axis=1, keepdims=True)
#     cm_percentage = np.divide(cm_sum.astype(np.float32), cm_sum, where=(cm_sum != 0)) * 100

#     # label_values = label_values[1:]
#     # label_values = label_values[1:]

#     vis.heatmap(cm_percentage, opts={'title': "Confusion matrix", 
#                           'marginbottom': 150,
#                           'marginleft': 150,
#                           'width': 500,
#                           'height': 500,
#                           'rownames': label_values, 'columnnames': label_values})
#     text += "Confusion matrix :\n"
#     text += str(cm_percentage)
#     text += "---\n"

#     if agregated:
#         text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
#                                                          np.std(accuracies)))
#     else:
#         text += "Accuracy : {:.03f}%\n".format(accuracy)
#     text += "---\n"

#     text += "F1 scores :\n"
#     if agregated:
#         for label, score, std in zip(label_values, F1_scores_mean,
#                                      F1_scores_std):
#             text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
#     else:
#         for label, score in zip(label_values, F1scores):
#             text += "\t{}: {:.03f}\n".format(label, score)
#     text += "---\n"

#     if agregated:
#         text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
#                                                       np.std(kappas)))
#     else:
#         text += "Kappa: {:.03f}\n".format(kappa)

#     vis.text(text.replace('\n', '<br/>'))
#     print(text)

"""-----------------------------------------New show results--------------------------------"""


def show_results(results, vis, label_values=None, aggregated=True, excel_path="metrics.xlsx"):
    # ... (existing code for calculating metrics) ...

    # For simplicity, assuming aggregated results
    # Accuracy, Kappa
    acc_mean, acc_std = np.mean([r["Accuracy"] for r in results]), np.std([r["Accuracy"] for r in results])
    kappas = [r["Kappa"] for r in results]
    kappa_mean, kappa_std = np.mean(kappas), np.std(kappas)

    # F1, Precision, Recall
    F1_scores = np.array([r["F1 scores"] for r in results])
    F1_scores_mean, F1_scores_std = np.mean(F1_scores, axis=0), np.std(F1_scores, axis=0)
    precisions = np.array([r["Precision"] for r in results])
    precision_mean, precision_std = np.mean(precisions, axis=0), np.std(precisions, axis=0)
    recalls = np.array([r["Recall"] for r in results])
    recall_mean, recall_std = np.mean(recalls, axis=0), np.std(recalls, axis=0)

    # Default class labels
    if label_values is None:
        label_values = [f"Class {i}" for i in range(len(F1_scores_mean))]

    # Indices for Water and Non-Water
    def get_index(name):
        return label_values.index(name) if name in label_values else None
    idx_water = get_index("Water")
    idx_nonwater = get_index("Non-Water")

    # Prepare table data with 2 decimal points
    data = {
        "Status": ["Aggregated"],
        "Accuracy (%)": [f"{acc_mean:.2f} ± {acc_std:.2f}"],
        "Kappa": [f"{kappa_mean:.2f} ± {kappa_std:.2f}"],
        "F1-Water": [f"{F1_scores_mean[idx_water]:.2f} ± {F1_scores_std[idx_water]:.2f}" if idx_water is not None else "0.00 ± 0.00"],
        "F1-Others": [f"{F1_scores_mean[idx_nonwater]:.2f} ± {F1_scores_std[idx_nonwater]:.2f}" if idx_nonwater is not None else "0.00 ± 0.00"],
        "Precision-Water": [f"{precision_mean[idx_water]:.2f} ± {precision_std[idx_water]:.2f}" if idx_water is not None else "0.00 ± 0.00"],
        "Precision-Others": [f"{precision_mean[idx_nonwater]:.2f} ± {precision_std[idx_nonwater]:.2f}" if idx_nonwater is not None else "0.00 ± 0.00"],
        "Recall-Water": [f"{recall_mean[idx_water]:.2f} ± {recall_std[idx_water]:.2f}" if idx_water is not None else "0.00 ± 0.00"],
        "Recall-Others": [f"{recall_mean[idx_nonwater]:.2f} ± {recall_std[idx_nonwater]:.2f}" if idx_nonwater is not None else "0.00 ± 0.00"]
    }

    # Save to Excel
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)
    print(f"Metrics table saved to {excel_path}")


# def show_results(results, vis, label_values=None, aggregated=True):
#     """
#     Display evaluation metrics using visdom including:
#     - Confusion Matrix (Percentage)
#     - Accuracy
#     - Precision, Recall, and F1 scores per class
#     - Kappa Coefficient

#     Args:
#         results: Dictionary or list of dicts containing evaluation metrics.
#         vis: Visualization tool (e.g., Visdom).
#         label_values (optional): List of class names.
#         aggregated (optional): Boolean flag for aggregated results.

#     Returns:
#         None (Displays results in Visdom and prints a formatted text summary)
#     """
#     text = ""

#     if aggregated:
#         accuracies = [r["Accuracy"] for r in results]
#         kappas = [r["Kappa"] for r in results]
#         F1_scores = np.array([r["F1 scores"] for r in results])
#         precisions = np.array([r["Precision"] for r in results])
#         recalls = np.array([r["Recall"] for r in results])        
        

#         F1_scores_mean = np.mean(F1_scores, axis=0)
#         F1_scores_std = np.std(F1_scores, axis=0)
#         precision_mean = np.mean(precisions, axis=0)
#         precision_std = np.std(precisions, axis=0)
#         recall_mean = np.mean(recalls, axis=0)
#         recall_std = np.std(recalls, axis=0)
        

#         cm = np.mean([r["Confusion matrix"] for r in results], axis=0)

#         text = "Aggregated Metrics:\n"
#     else:
#         cm = results["Confusion matrix"]
#         accuracy = results["Accuracy"]
#         F1_scores = results["F1 scores"]
#         precision = results["Precision"]
#         recall = results["Recall"]
#         kappa = results["Kappa"]    

    
#     cm_filtered, label_values_filtered = filter_confusion_matrix(cm, label_values)

#     # Generate default class labels if not provided
#     if label_values_filtered is None:
#         label_values_filtered = [f"Class {i}" for i in range(len(cm_filtered))]

#     # Normalize the confusion matrix (row-wise)
#     cm_sum = cm_filtered.sum(axis=1, keepdims=True)
#     cm_percentage = np.divide(cm_filtered, cm_sum, where=(cm_sum > 0), out=np.zeros_like(cm_filtered, dtype=float)) * 100

#     vis.heatmap(
#         cm_percentage, 
#         opts={
#             'title': "Confusion Matrix (Percentage)", 
#             'rownames': label_values_filtered, 
#             'columnnames': label_values_filtered,
#             'colormap': 'Viridis'
#         }
#     )

#     text += "Confusion Matrix (Percentage):\n"
#     text += str(np.round(cm_percentage, 2)) + "\n"
#     text += "---\n"

#     # Display Accuracy
#     if aggregated:
#         text += "Accuracy: {:.3f}% ± {:.3f}\n".format(np.mean(accuracies), np.std(accuracies))
#     else:
#         text += "Accuracy: {:.3f}%\n".format(accuracy)
#     text += "---\n"

#     # Display Precision, Recall, and F1 scores per class
#     if aggregated:
#         text += "Precision (Mean ± Std):\n"
#         for label, mean, std in zip(label_values, precision_mean, precision_std):
#             text += "\t{}: {:.3f} ± {:.03f}\n".format(label, mean, std)

#         text += "Recall (Mean ± Std):\n"
#         for label, mean, std in zip(label_values, recall_mean, recall_std):
#             text += "\t{}: {:.3f} ± {:.03f}\n".format(label, mean, std)

#         text += "F1 Scores (Mean ± Std):\n"
#         for label, mean, std in zip(label_values, F1_scores_mean, F1_scores_std):
#             text += "\t{}: {:.03f} ± {:.03f}\n".format(label, mean, std)
#     else:
#         text += "Precision:\n"
#         for label, p in zip(label_values, precision):
#             text += "\t{}: {:.3f}\n".format(label, p)

#         text += "Recall:\n"
#         for label, r in zip(label_values, recall):
#             text += "\t{}: {:.3f}\n".format(label, r)

#         text += "F1 Scores:\n"
#         for label, score in zip(label_values, F1_scores):
#             text += "\t{}: {:.3f}\n".format(label, score)

#     text += "---\n"

#     if aggregated:
#         text += "Kappa: {:.3f} ± {:.3f}\n".format(np.mean(kappas), np.std(kappas))
#     else:
#         text += "Kappa: {:.3f}\n".format(kappa)

#     vis.text(text.replace('\n', '<br/>'))
#     print(text)

"""-----------------------------------------New show results--------------------------------"""

"""---------------------------------------cm filtering-----------------------------"""
def filter_confusion_matrix(cm, label_values, ignored_labels=["Undefined"]):
    """
    Removes specified labels (e.g., "Undefined") from the confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        label_values (list): List of class names.
        ignored_labels (list): Labels to remove.

    Returns:
        tuple: (filtered_cm, filtered_labels)
    """
    # Find indices of labels to remove
    remove_indices = [i for i, label in enumerate(label_values) if label in ignored_labels]

    # Keep only valid indices
    keep_indices = [i for i in range(len(label_values)) if i not in remove_indices]

    # Filter confusion matrix and labels
    cm_filtered = cm[np.ix_(keep_indices, keep_indices)]
    labels_filtered = [label_values[i] for i in keep_indices]

    return cm_filtered, labels_filtered

"""---------------------------------------cm filtering-----------------------------"""

def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
       test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


