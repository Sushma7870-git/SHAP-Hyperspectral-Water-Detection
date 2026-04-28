# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
import cv2
# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# Visualization
import seaborn as sns
import visdom
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import shap
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot
import time
from captum.attr import IntegratedGradients
import os
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions,plot_image_with_legend, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
import argparse
dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=-1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=0.1,
                    help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                               FOLDER)
# Number of classes
# N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
N_CLASSES = len(LABEL_VALUES)
print("LABEL_VALUES----------",LABEL_VALUES )
print("class----------",N_CLASSES )
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                       ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w,:h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
	# Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    if MODEL == 'SVM_grid':
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SVM':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'nearest':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.neighbors.KNeighborsClassifier(weights='distance')
        clf = sklearn.model_selection.GridSearchCV(clf, {'n_neighbors': [1, 3, 5, 10, 20]}, verbose=5, n_jobs=4)
        clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    else:
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            print("weights-----------", weights)
            hyperparams['weights'] = torch.from_numpy(weights)
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        # Split train set in train/val
        train_gt, val_gt = sample_gt(train_gt, 0.80, mode='random')
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       #pin_memory=hyperparams['device'],
                                       shuffle=True)
        val_dataset = HyperX(img, val_gt, **hyperparams)
        val_loader = data.DataLoader(val_dataset,
                                     #pin_memory=hyperparams['device'],
                                     batch_size=hyperparams['batch_size'])

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            #summary(model.to(hyperparams['device']), input.size()[1:], device=hyperparams['device'])
            summary(model.to(hyperparams['device']), input.size()[1:])

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'], val_loader=val_loader,
                  display=viz)
            end = time.time()
            elapsed_minutes = (end - start) / 60
            peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0
    color_prediction = convert_to_color(prediction)    
    display_predictions(color_prediction, viz,caption="Prediction vs. test ground truth")  
    
    class_map = {
    0: 'Undefined',
    1: 'Water',
    2: 'Non-Water'
    }

    # Plot the predicted image with class legends
    plot_image_with_legend(prediction, class_map)
    results.append(run_results)    
    show_results(results, viz, label_values=LABEL_VALUES, aggregated=True, excel_path="metric_tables\\aggregated_metrics_before.xlsx")    

    excel_path = "metric_tables\\aggregated_metrics_before.xlsx"
    # Load the Excel file
    df = pd.read_excel(excel_path)
    # Add new columns (same value for aggregated row)
    df["Training Time (min)"] = f"{elapsed_minutes:.2f}"
    df["Peak GPU Memory (GB)"] = f"{peak_gpu_gb:.2f}"
    # Save back
    df.to_excel(excel_path, index=False)

    print(f"Training time: {elapsed_minutes:.2f} min")
    print(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")
    print("✓ Training stats added to aggregated_metrics_before.xlsx")

    
    """-------------------SHAP ANALYSIS WITH ADAPTIVE BAND SELECTION----------------"""

    

    # ========================================================================
    # STEP 1: Compute SHAP Values for ALL Classes
    # ========================================================================

    def select_important_bands_multiclass(model, test_loader, hyperparams, 
                                        n_bands_to_keep=None, 
                                        class_weights=None):
        """
        Select most important bands considering ALL classes.
        If n_bands_to_keep is None, returns all bands sorted by importance.
        """
        
        # Get a batch for SHAP computation
        for inputs, _ in test_loader:
            break
        
        inputs = inputs.to(hyperparams['device'])
        inputs.requires_grad = True
        model.eval()
        
        # Compute SHAP values
        print("Computing SHAP values for all classes...")
        explainer = shap.GradientExplainer(model, inputs)
        shap_values = explainer.shap_values(inputs)
        
        # Extract center pixel: [B, C, Bands, H, W, n_classes] -> [B, Bands, n_classes]
        center_h, center_w = 2, 2
        shap_values_np = shap_values[:, 0, :, center_h, center_w, :]
        
        n_classes = shap_values_np.shape[2]
        n_bands = shap_values_np.shape[1]
        
        # Default: equal weights for all classes, but reduce "Undefined" class weight
        if class_weights is None:
            class_weights = {0: 0.05, 1: 0.475, 2: 0.475}  # Undefined, Water, Non-water
        
        print(f"\nClass weights: {class_weights}")
        
        # Compute importance for each class
        importance_per_class = {}
        raw_shap_per_class = {}  # NEW: Store raw SHAP values with signs
        class_labels = ['Undefined', 'Water', 'Non-water']
        
        for cls in range(n_classes):
            # Store raw SHAP values (with positive/negative signs)
            raw_shap = shap_values_np[:, :, cls].mean(axis=0)
            raw_shap_per_class[cls] = raw_shap
            
            # Compute importance using absolute values
            importance = np.abs(shap_values_np[:, :, cls]).mean(axis=0)
            importance_per_class[cls] = importance
            print(f"Class {cls} ({class_labels[cls]}): "
                f"Mean importance = {importance.mean():.4f}, "
                f"Max = {importance.max():.4f}, "
                f"Mean raw SHAP = {raw_shap.mean():.4f}")
        
        # Combine importance across classes with weights
        combined_importance = np.zeros(n_bands)
        for cls in range(n_classes):
            weight = class_weights.get(cls, 1.0 / n_classes)
            combined_importance += weight * importance_per_class[cls]
        
        # Sort bands by importance
        sorted_band_indices = np.argsort(combined_importance)[::-1]
        sorted_importance = combined_importance[sorted_band_indices]
        
        if n_bands_to_keep is not None:
            # Select top N bands
            selected_bands = np.sort(sorted_band_indices[:n_bands_to_keep])
        else:
            # Return all bands sorted
            selected_bands = sorted_band_indices
        
        return selected_bands, importance_per_class, combined_importance, sorted_band_indices, sorted_importance, raw_shap_per_class


    # ========================================================================
    # STEP 2: Thresholding Methods for Automatic Band Selection
    # ========================================================================

    def select_bands_by_threshold(combined_importance, sorted_band_indices, sorted_importance, 
                                method='cumulative', threshold=0.95, min_bands=5, max_bands=None):
        """
        Select bands using various thresholding methods.
        
        Args:
            combined_importance: Array of importance scores for all bands
            sorted_band_indices: Band indices sorted by importance (descending)
            sorted_importance: Importance scores sorted (descending)
            method: Thresholding method
                - 'cumulative': Keep bands until cumulative importance reaches threshold (e.g., 95%)
                - 'elbow': Use elbow method to find optimal cutoff
                - 'percentage': Keep bands with importance >= threshold * max_importance
                - 'std': Keep bands with importance >= mean + threshold * std
                - 'gap': Keep bands until a large gap in importance is detected
            threshold: Threshold value (meaning depends on method)
            min_bands: Minimum number of bands to keep
            max_bands: Maximum number of bands to keep
        
        Returns:
            selected_bands: Array of selected band indices (sorted)
            n_selected: Number of selected bands
            selection_info: Dict with additional information about the selection
        """
        
        n_bands = len(combined_importance)
        if max_bands is None:
            max_bands = n_bands
        
        print(f"\n{'='*60}")
        print(f"BAND SELECTION METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        selection_info = {'method': method, 'threshold': threshold}
        
        if method == 'cumulative':
            # Keep bands until cumulative importance reaches threshold
            cumsum = np.cumsum(sorted_importance)
            total = cumsum[-1]
            cumulative_ratio = cumsum / total
            
            n_selected = np.searchsorted(cumulative_ratio, threshold) + 1
            n_selected = max(min_bands, min(n_selected, max_bands))
            
            selection_info.update({
                'total_importance': total,
                'cumulative_importance': cumsum[n_selected-1],
                'cumulative_ratio': cumulative_ratio[n_selected-1]
            })
            
            print(f"Threshold: {threshold*100:.1f}% of total importance")
            print(f"Total importance: {total:.4f}")
            print(f"Cumulative importance at {n_selected} bands: {cumsum[n_selected-1]:.4f} ({cumulative_ratio[n_selected-1]*100:.2f}%)")
            
        elif method == 'elbow':
            # Use elbow method (point of maximum curvature)
            from scipy.ndimage import gaussian_filter1d
            
            # Smooth the curve
            smoothed = gaussian_filter1d(sorted_importance, sigma=2)
            
            # Compute second derivative
            second_deriv = np.diff(smoothed, n=2)
            
            # Find elbow (maximum of second derivative)
            n_selected = np.argmax(np.abs(second_deriv)) + 2  # +2 due to diff
            n_selected = max(min_bands, min(n_selected, max_bands))
            
            selection_info.update({
                'elbow_position': n_selected,
                'importance_at_elbow': sorted_importance[n_selected-1]
            })
            
            print(f"Elbow detected at band {n_selected}")
            print(f"Importance at elbow: {sorted_importance[n_selected-1]:.4f}")
            
        elif method == 'percentage':
            # Keep bands with importance >= threshold * max_importance
            max_imp = sorted_importance[0]
            cutoff = threshold * max_imp
            
            n_selected = np.sum(sorted_importance >= cutoff)
            n_selected = max(min_bands, min(n_selected, max_bands))
            
            selection_info.update({
                'max_importance': max_imp,
                'cutoff_value': cutoff
            })
            
            print(f"Threshold: {threshold*100:.1f}% of maximum importance")
            print(f"Maximum importance: {max_imp:.4f}")
            print(f"Cutoff value: {cutoff:.4f}")
            print(f"Bands above threshold: {n_selected}")
            
        elif method == 'std':
            # Keep bands with importance >= mean + threshold * std
            mean_imp = combined_importance.mean()
            std_imp = combined_importance.std()
            cutoff = mean_imp + threshold * std_imp
            
            n_selected = np.sum(sorted_importance >= cutoff)
            n_selected = max(min_bands, min(n_selected, max_bands))
            
            selection_info.update({
                'mean_importance': mean_imp,
                'std_importance': std_imp,
                'cutoff_value': cutoff
            })
            
            print(f"Mean importance: {mean_imp:.4f}")
            print(f"Std importance: {std_imp:.4f}")
            print(f"Threshold: mean + {threshold:.1f} * std = {cutoff:.4f}")
            print(f"Bands above threshold: {n_selected}")
            
        elif method == 'gap':
            # Detect large gap in importance
            diffs = np.diff(sorted_importance)
            
            # Find the largest gap
            median_gap = np.median(np.abs(diffs))
            large_gaps = np.where(np.abs(diffs) > threshold * median_gap)[0]
            
            if len(large_gaps) > 0:
                n_selected = large_gaps[0] + 1  # +1 to include the band before the gap
            else:
                n_selected = len(sorted_importance) // 2  # Default to half if no gap found
            
            n_selected = max(min_bands, min(n_selected, max_bands))
            
            selection_info.update({
                'median_gap': median_gap,
                'gap_threshold': threshold * median_gap,
                'gap_position': n_selected,
                'gap_value': np.abs(diffs[n_selected-1]) if n_selected > 0 else 0
            })
            
            print(f"Median gap: {median_gap:.6f}")
            print(f"Threshold: {threshold:.1f} * median_gap = {threshold * median_gap:.6f}")
            print(f"Largest gap found at position {n_selected}")
            print(f"Gap value: {selection_info['gap_value']:.6f}")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selected_bands = np.sort(sorted_band_indices[:n_selected])
        selection_info['n_selected'] = n_selected
        selection_info['selected_bands'] = selected_bands
        
        print(f"\nSelected {n_selected} bands ({n_selected/n_bands*100:.1f}% of total)")
        print(f"Band reduction: {n_bands} → {n_selected} ({(n_bands-n_selected)/n_bands*100:.1f}% reduction)")
        
        return selected_bands, n_selected, selection_info


    # ========================================================================
    # STEP 3: Visualize All Thresholding Methods
    # ========================================================================

    def visualize_threshold_methods(combined_importance, sorted_band_indices, sorted_importance):
        """
        Visualize different thresholding methods to help choose the best one.
        """
        
        n_bands = len(combined_importance)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Method 1: Cumulative Importance
        ax = axes[0]
        cumsum = np.cumsum(sorted_importance)
        cumulative_ratio = cumsum / cumsum[-1]
        
        ax.plot(range(1, n_bands+1), cumulative_ratio * 100, 'b-', linewidth=2)
        ax.axhline(y=90, color='r', linestyle='--', label='90% threshold', alpha=0.7)
        ax.axhline(y=95, color='orange', linestyle='--', label='95% threshold', alpha=0.7)
        ax.axhline(y=99, color='green', linestyle='--', label='99% threshold', alpha=0.7)
        
        n_90 = np.searchsorted(cumulative_ratio, 0.90) + 1
        n_95 = np.searchsorted(cumulative_ratio, 0.95) + 1
        n_99 = np.searchsorted(cumulative_ratio, 0.99) + 1
        
        ax.scatter([n_90, n_95, n_99], [90, 95, 99], s=100, c=['r', 'orange', 'green'], zorder=5)
        ax.text(n_90, 88, f'{n_90}', ha='center', fontsize=9, fontweight='bold')
        ax.text(n_95, 93, f'{n_95}', ha='center', fontsize=9, fontweight='bold')
        ax.text(n_99, 97, f'{n_99}', ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Number of Bands', fontsize=11)
        ax.set_ylabel('Cumulative Importance (%)', fontsize=11)
        ax.set_title(f'Cumulative Method\n90%→{n_90}, 95%→{n_95}, 99%→{n_99} bands', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Method 2: Elbow Method
        ax = axes[1]
        ax.plot(range(1, n_bands+1), sorted_importance, 'b-', linewidth=2)
        
        # Find elbow
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(sorted_importance, sigma=2)
        second_deriv = np.diff(smoothed, n=2)
        elbow_idx = np.argmax(np.abs(second_deriv)) + 2
        
        ax.axvline(x=elbow_idx, color='r', linestyle='--', linewidth=2, label=f'Elbow at {elbow_idx}', alpha=0.7)
        ax.scatter([elbow_idx], [sorted_importance[elbow_idx-1]], s=150, c='r', zorder=5, marker='*')
        ax.set_xlabel('Band Rank', fontsize=11)
        ax.set_ylabel('Importance', fontsize=11)
        ax.set_title(f'Elbow Method\nElbow at {elbow_idx} bands', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Method 3: Percentage of Max
        ax = axes[2]
        max_imp = sorted_importance[0]
        
        thresholds = [(0.1, 'r', '10%'), (0.2, 'orange', '20%'), (0.3, 'green', '30%')]
        for pct, color, label in thresholds:
            cutoff = pct * max_imp
            n_bands_pct = np.sum(sorted_importance >= cutoff)
            ax.axhline(y=cutoff, color=color, linestyle='--', label=f'{label}: {n_bands_pct} bands', alpha=0.7)
        
        ax.plot(range(1, n_bands+1), sorted_importance, 'b-', linewidth=2)
        ax.set_xlabel('Band Rank', fontsize=11)
        ax.set_ylabel('Importance', fontsize=11)
        ax.set_title('Percentage of Max Method', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Method 4: Standard Deviation
        ax = axes[3]
        mean_imp = combined_importance.mean()
        std_imp = combined_importance.std()
        
        std_thresholds = [(1.0, 'r', '1σ'), (1.5, 'orange', '1.5σ'), (2.0, 'green', '2σ')]
        for mult, color, label in std_thresholds:
            cutoff = mean_imp + mult * std_imp
            n_bands_std = np.sum(sorted_importance >= cutoff)
            ax.axhline(y=cutoff, color=color, linestyle='--', label=f'μ+{label}: {n_bands_std} bands', alpha=0.7)
        
        ax.plot(range(1, n_bands+1), sorted_importance, 'b-', linewidth=2)
        ax.axhline(y=mean_imp, color='gray', linestyle=':', linewidth=2, label=f'Mean', alpha=0.7)
        ax.set_xlabel('Band Rank', fontsize=11)
        ax.set_ylabel('Importance', fontsize=11)
        ax.set_title('Standard Deviation Method', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Method 5: Gap Detection
        ax = axes[4]
        diffs = np.abs(np.diff(sorted_importance))
        median_gap = np.median(diffs)
        
        ax.plot(range(1, len(diffs)+1), diffs, 'b-', linewidth=2)
        
        gap_thresholds = [(2.0, 'r', '2×'), (3.0, 'orange', '3×'), (5.0, 'green', '5×')]
        for mult, color, label in gap_thresholds:
            threshold = mult * median_gap
            large_gaps = np.where(diffs > threshold)[0]
            if len(large_gaps) > 0:
                first_gap = large_gaps[0] + 1
                ax.scatter([first_gap], [diffs[large_gaps[0]]], s=100, c=color, zorder=5,
                        label=f'{label}median: {first_gap} bands')
        
        ax.axhline(y=median_gap, color='gray', linestyle=':', linewidth=2, label=f'Median', alpha=0.7)
        ax.set_xlabel('Band Rank', fontsize=11)
        ax.set_ylabel('Importance Gap', fontsize=11)
        ax.set_title('Gap Detection Method', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Method 6: All Bands Importance (reference)
        ax = axes[5]
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, n_bands))
        ax.bar(range(n_bands), combined_importance[sorted_band_indices], color=colors_gradient, alpha=0.7, width=1.0)
        ax.set_xlabel('Band Rank', fontsize=11)
        ax.set_ylabel('Importance', fontsize=11)
        ax.set_title('All Bands Sorted by Importance', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    # ========================================================================
    # STEP 4: Visualize Selected Bands (Multi-class view)
    # ========================================================================

    def visualize_band_selection_multiclass(importance_per_class, combined_importance, 
                                            selected_bands, class_labels):
        """
        Create comprehensive visualizations of band importance across classes.
        """
        n_bands = len(combined_importance)
        n_classes = len(importance_per_class)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Combined importance with selected bands highlighted
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(range(n_bands), combined_importance, 'b-', linewidth=1.5, alpha=0.6, 
                label='Combined Importance')
        ax1.scatter(selected_bands, combined_importance[selected_bands], 
                    color='red', s=100, zorder=5, label=f'Selected Bands (n={len(selected_bands)})')
        ax1.set_xlabel('Band Index', fontsize=12)
        ax1.set_ylabel('Combined Importance', fontsize=12)
        ax1.set_title('Band Selection Based on Multi-Class Importance', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2 & 3: Individual class importance (Water and Non-water)
        colors = ['skyblue', 'lightgreen', 'orange']
        for i, cls in enumerate([1, 2]):  # Water and Non-water only
            row = 1 if cls == 1 else 1
            col = 0 if cls == 1 else 1
            
            ax = fig.add_subplot(gs[row, col])
            importance = importance_per_class[cls]
            
            ax.plot(range(n_bands), importance, color=colors[cls], linewidth=1.5, alpha=0.6)
            ax.scatter(selected_bands, importance[selected_bands], 
                    color='red', s=80, zorder=5, alpha=0.7)
            ax.set_xlabel('Band Index', fontsize=11)
            ax.set_ylabel('Importance', fontsize=11)
            ax.set_title(f'{class_labels[cls]} Class Importance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Stacked bar chart of selected bands
        ax4 = fig.add_subplot(gs[2, :])
        
        n_show = min(30, len(selected_bands))
        x_pos = np.arange(n_show)
        
        # Stack importance from each class
        bottom = np.zeros(n_show)
        for cls in range(n_classes):
            importance = importance_per_class[cls][selected_bands[:n_show]]
            ax4.bar(x_pos, importance, bottom=bottom, label=class_labels[cls], 
                    color=colors[cls], alpha=0.7)
            bottom += importance
        
        ax4.set_xlabel('Selected Band Index', fontsize=12)
        ax4.set_ylabel('Stacked Importance', fontsize=12)
        ax4.set_title(f'Class-wise Contribution to Top {n_show} Selected Bands', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(selected_bands[:n_show], rotation=45, ha='right')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional heatmap visualization
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create matrix: [n_classes x selected_bands]
        importance_matrix = np.zeros((n_classes, len(selected_bands)))
        for cls in range(n_classes):
            importance_matrix[cls, :] = importance_per_class[cls][selected_bands]
        
        im = ax.imshow(importance_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_labels, fontsize=11)
        ax.set_xlabel('Selected Band Rank', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title('Importance Heatmap: Classes vs Selected Bands', fontsize=14, fontweight='bold')
        
        # Show band indices on x-axis
        n_show = min(30, len(selected_bands))
        tick_positions = np.linspace(0, len(selected_bands)-1, n_show, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(selected_bands[tick_positions], rotation=45, ha='right')
        
        plt.colorbar(im, ax=ax, label='Importance Score')
        plt.tight_layout()
        plt.show()


    # ========================================================================
    # MAIN WORKFLOW - BAND SELECTION ONLY
    # ========================================================================

    

    # Load your data
    test_dataset = HyperX(img, val_gt, **hyperparams)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    # Define class labels
    class_labels = ['Undefined', 'Water', 'Non-water']

    # Step 1: Compute SHAP values and importance scores
    

    class_weights = {0: 0.05, 1: 0.475, 2: 0.475}
    selected_bands_all, importance_per_class, combined_importance, sorted_band_indices, sorted_importance, raw_shap_per_class = \
        select_important_bands_multiclass(model, test_loader, hyperparams, 
                                        n_bands_to_keep=None, 
                                        class_weights=class_weights)

    # Step 2: Visualize all thresholding methods
    
    visualize_threshold_methods(combined_importance, sorted_band_indices, sorted_importance)

    # Step 3: Select bands using chosen method
    

    # CHOOSE YOUR METHOD HERE - Set method_choice to 1-6:
    method_choice = 2  # Change this value to select different methods

    # Method configurations
    methods_config = {
        1: {'method': 'cumulative', 'threshold': 0.90, 'description': 'Cumulative 90%'},
        2: {'method': 'cumulative', 'threshold': 0.95, 'description': 'Cumulative 95% [RECOMMENDED]'},
        3: {'method': 'elbow', 'threshold': None, 'description': 'Automatic Elbow Detection'},
        4: {'method': 'percentage', 'threshold': 0.2, 'description': 'Percentage of Max (20%)'},
        5: {'method': 'std', 'threshold': 1.5, 'description': 'Standard Deviation (mean + 1.5σ)'},
        6: {'method': 'gap', 'threshold': 3.0, 'description': 'Gap Detection (3× median)'}
    }

    # Validate method_choice
    if method_choice not in methods_config:
        raise ValueError(f"method_choice must be between 1 and 6, got {method_choice}")

    selected_config = methods_config[method_choice]
    print(f"\nUsing Method {method_choice}: {selected_config['description']}")

    final_bands, n_final, selection_info = select_bands_by_threshold(
        combined_importance, sorted_band_indices, sorted_importance,
        method=selected_config['method'],
        threshold=selected_config['threshold'],
        min_bands=10,           # Minimum bands to keep
        max_bands=100            # Maximum bands to keep
    )

    # Step 4: Visualize selected bands
    

    visualize_band_selection_multiclass(importance_per_class, combined_importance, 
                                        final_bands, class_labels)

    # Step 5: Detailed breakdown of selected bands
    
    for rank, band_idx in enumerate(final_bands[:20], 1):  # Show top 20
        combined_imp = combined_importance[band_idx]
        water_imp = importance_per_class[1][band_idx]
        nonwater_imp = importance_per_class[2][band_idx]
        
        if water_imp > nonwater_imp:
            dominant = "Water"
        else:
            dominant = "Non-water"
        
       
    #  Save as CSV for easy viewing in Excel/Pandas
    

    df_importance = pd.DataFrame({
        'Band_Index': np.arange(len(combined_importance)),
        'Combined_Importance': combined_importance,
        'Undefined_Importance': importance_per_class[0],
        'Water_Importance': importance_per_class[1],
        'Non-Water_Importance': importance_per_class[2],
        'Raw_SHAP_Undefined': raw_shap_per_class[0],  # NEW: Raw SHAP with signs
        'Raw_SHAP_Water': raw_shap_per_class[1],
        'Raw_SHAP_Non-Water': raw_shap_per_class[2],
        'Is_Selected': [1 if i in final_bands else 0 for i in range(len(combined_importance))],
        'Rank': [np.where(sorted_band_indices == i)[0][0] + 1 for i in range(len(combined_importance))]
    })

    # Sort by combined importance for easy viewing
    df_importance_sorted = df_importance.sort_values('Combined_Importance', ascending=False)
    df_importance_sorted.to_csv('metric_tables\\all_bands_shap_importance.csv', index=False)
    print(f"✓ All band SHAP importance saved to: 'metric_tables\\all_bands_shap_importance.csv'")

    # Save a detailed analysis CSV with statistics
    df_stats = pd.DataFrame({
        'Band_Index': final_bands,
        'Rank': np.arange(1, len(final_bands) + 1),
        'Combined_Importance': combined_importance[final_bands],
        'Undefined_Importance': importance_per_class[0][final_bands],
        'Water_Importance': importance_per_class[1][final_bands],
        'Non-Water_Importance': importance_per_class[2][final_bands],
        'Raw_SHAP_Undefined': raw_shap_per_class[0][final_bands],  # NEW: Raw SHAP with signs
        'Raw_SHAP_Water': raw_shap_per_class[1][final_bands],
        'Raw_SHAP_Non-Water': raw_shap_per_class[2][final_bands],
        'Dominant_Class': ['Water' if importance_per_class[1][b] > importance_per_class[2][b] 
                        else 'Non-water' for b in final_bands]
    })
    df_stats.to_csv('metric_tables\\selected_bands_detailed.csv', index=False)
    print(f"✓ Selected bands detailed analysis saved to: 'metric_tables\\selected_bands_detailed.csv'")

    """
    RETRAIN MODEL WITH REDUCED BANDS (95% SHAP CSV)
    """

    # # ========================================================================
    # # STEP 1: Load Selected Bands from CSV
    # # ========================================================================

    
    bands_df = pd.read_csv(r'metric_tables\\selected_bands_detailed.csv')
    # Extract the column by name
    selected_bands = bands_df['Band_Index'].values.astype(int)

    print(f"✓ Loaded {len(selected_bands)} selected bands")
    print(f"Selected band indices: {selected_bands}")

    # Safety check
    assert selected_bands.max() < img.shape[2], "Band index exceeds available bands!"
    assert selected_bands.min() >= 0, "Band index must be non-negative!"
    # ========================================================================
    # STEP 2: Reduce the Hyperspectral Image
    # ========================================================================
    img_reduced = img[:, :, selected_bands]
    # ========================================================================
    # STEP 3: Update Hyperparameters
    # ========================================================================

    original_n_bands = hyperparams['n_bands']   

    hyperparams['n_bands'] = len(selected_bands)

    # ========================================================================
    # STEP 4: Retrain Model with Reduced Bands
    # ========================================================================

    # Initialize new model with updated hyperparameters
    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

    # Split train set into train/validation
    train_gt, val_gt = sample_gt(train_gt, 0.80, mode='random')

    # Create datasets using REDUCED IMAGE
    train_dataset = HyperX(img_reduced, train_gt, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True
    )

    val_dataset = HyperX(img_reduced, val_gt, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=hyperparams['batch_size']
    )

    
    with torch.no_grad():
        for input, _ in train_loader:
            break
        summary(model.to(hyperparams['device']), input.size()[1:])

    # Load checkpoint if provided
    if CHECKPOINT is not None:
        print(f"\nLoading checkpoint from: {CHECKPOINT}")
        model.load_state_dict(torch.load(CHECKPOINT))

    # ================== TRAIN ==================

    try:
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        train(
            model,
            optimizer,
            loss,
            train_loader,
            hyperparams['epoch'],
            scheduler=hyperparams['scheduler'],
            device=hyperparams['device'],
            supervision=hyperparams['supervision'],
            val_loader=val_loader,
            display=viz
        )

        end = time.time()
        elapsed_minutes = (end - start) / 60
        peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print("TRAINING COMPLETED")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # ========================================================================
    # STEP 5: Test with Reduced Bands
    # ========================================================================

    probabilities = test(model, img_reduced, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)

    # ========================================================================
    # STEP 6: Evaluate Results
    # ========================================================================

    run_results = metrics(
        prediction,
        test_gt,
        ignored_labels=hyperparams['ignored_labels'],
        n_classes=N_CLASSES
    )

    # Mask ignored labels
    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    # Visualize predictions
    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        caption="Prediction (Reduced Bands - 95%) vs. Test Ground Truth"
    )
    # Class map
    class_map = {
        0: 'Undefined',
        1: 'Water',
        2: 'Non-Water'
    }

    results.append(run_results)


    # """--------------Band Importance---------------"""

if N_RUNS == 1:
    show_results(results, viz, label_values=LABEL_VALUES, aggregated=True, excel_path="metric_tables\\aggregated_metrics_after.xlsx")

    excel_path = "aggregated_metrics_after.xlsx"

    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Add new columns (same value for aggregated row)
    df["Training Time (min)"] = f"{elapsed_minutes:.2f}"
    df["Peak GPU Memory (GB)"] = f"{peak_gpu_gb:.2f}"

    # Save back
    df.to_excel(excel_path, index=False)
    print(f"Training time: {elapsed_minutes:.2f} min")
    print(f"Peak GPU memory: {peak_gpu_gb:.2f} GB")
    print("✓ Training stats added to aggregated_metrics_after.xlsx")
# else:
#     # show_results(results, viz, label_values=LABEL_VALUES, aggregated=True, excel_path="aggregated_metrics_before.xlsx")
#     show_results(run_results, viz, label_values=LABEL_VALUES,aggregated=True)