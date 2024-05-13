import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import seaborn as sns
import torch
import math
from .metrics import (
    pearsonr, r2_score, max_error, mean_absolute_percentage_error, median_absolute_error, mean_absolute_error, mean_squared_error,
    f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, get_rm2, get_cindex, roc_auc_score, auc,
    get_metrics_reg, get_metrics_cls, precision_recall_curve
    )
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
from cadd_lead_optimisation.utils.helpers import io as helper_io
import warnings

# Setting Seaborn style for all plots
sns.set(style="whitegrid")

# Ignoring Warnings
warnings.filterwarnings("ignore")

# Plotting Functions
def plot_regression(y_true, y_pred, value_obj, value_name, value_abbrev, title_addition=None, filepath=None, single_val=True, fig_size=(10, 6), return_ax=False):
    # Checking for multiple or single values
    multiple_values = None
    if value_abbrev == "R-VALUE":
        single_value = value_obj(y_true.flatten(), y_pred.flatten())[0]
    else:
        single_value = value_obj(y_true, y_pred)
    if not single_val:
        multiple_values = value_obj(y_true, y_pred, multioutput="raw_values")

    print(f"Single Val: {single_val}, Multiple Vals: {multiple_values}")
    
    # Getting plots
    fig, ax1 = plt.subplots(figsize=fig_size)
    sns.scatterplot(x=y_true, y=y_pred, ax=ax1, color='blue', label='Predictions')
    ax1.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'k--', lw=2)
    ax2 = None
    
    if multiple_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(multiple_values)), multiple_values, 'r-')
        ax2.set_ylabel(f"{value_name} Values", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot Mods
    if title_addition:    
        title = title_addition + f" True vs Predicted Values with {value_name}"
    else:
        title = f"True vs Predicted Values with {value_name}"
        
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(f"{title} ({value_abbrev}: {single_value:.3f})")
    
    # Saving the figure with adjusted layout
    
    if filepath:
        new_file_name = f"{filepath.stem}_{value_name.replace(' ', '_')}.png"
        new_file_path = filepath.parent
        helper_io.create_folder(new_file_path)
        new_file_path = new_file_path / new_file_name
        plt.savefig(new_file_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    if return_ax:
        if multiple_values is not None:
            return ax1, ax2
        return ax1, None
    return None

# Need to modify the below to include the filepath related things.
def plot_classification(y_true, y_pred, value_obj, value_name, value_abbrev, transform=torch.sigmoid, threshold=0.5, single_val=True, fig_size=(10, 6), return_ax=False):
    # Converting y_pred and y_true
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_pred = transform(y_pred) if transform else y_pred
    y_pred_lbl = (y_pred >= threshold).type(torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32) if not isinstance(y_true, torch.Tensor) else y_true

    # Convert y_pred and y_true into numpy arrays
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_lbl_np = y_pred_lbl.detach().cpu().numpy()
    
    # Checking for multiple or single values
    multiple_values = None
    if value_abbrev == "PS":
        single_value = value_obj(y_true_np.ravel(), y_pred_lbl_np.ravel(), zero_division=0)
    elif value_abbrev == "ROC_AUC":
        try:
            single_value = value_obj(y_true_np.ravel(), y_pred_np.ravel())
        except:
            single_value = np.nan
    elif value_abbrev == "AUC":
        try:
            precision_list, recall_list, _ = precision_recall_curve(y_true_np.ravel(), y_pred_np.ravel())
            single_value = value_obj(recall_list, precision_list)
        except:
            single_value = np.nan
    else:
        single_value = value_obj(y_true_np.ravel(), y_pred_lbl_np.ravel())
        
    if not single_val:
        if value_abbrev == "PS":
            multiple_values = value_obj(y_true_np, y_pred_lbl_np, average=None, zero_division=0)
        else:
            multiple_values = value_obj(y_true_np, y_pred_lbl_np, average=None)

    print(f"y_true: {y_true}\n")
    print(f"y_pred: {y_pred}\n")
    print(f"y_pred_lbl: {y_pred_lbl}\n\n\n")
    
    print(f"y_true_np: {y_true_np}\n")
    print(f"y_pred_np: {y_pred_np}\n")
    print(f"y_pred_lbl_np: {y_pred_lbl_np}\n")
    
    print(f"single_value: {single_value}\n")
    print(f"multiple_values: {multiple_values}\n\n\n")
    
    # Getting plots
    fig, ax1 = plt.subplots(figsize=fig_size)
    sns.scatterplot(x=y_true, y=y_pred, ax=ax1, color='blue', label='Predictions')
    ax1.plot([min(y_pred_np), max(y_pred_np)], [min(y_pred_np), max(y_pred_np)], 'k--', lw=2)
    ax2 = None
    
    if multiple_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(multiple_values)), multiple_values, 'r-')
        ax2.set_ylabel(f"{value_name} Values", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot Mods
    title = f"True vs Predicted Values with {value_name}"
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(f"{title} ({value_abbrev}: {single_value:.3f})")
        
    if return_ax:
        return ax1, ax2  # Always return a tuple of two elements
    return ax1, None  # If return_ax is False, return ax1 and None

# Making Plotting Factory (to hold all plotting functions)
def get_plot_method_reg(plot_method, y_true, y_pred, title_addition=None, filepath=None, with_rm2=True, with_ci=True, return_ax=True):
    plot_object_dict_kwargs = {
        "mse": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=mean_squared_error, value_name="Mean Squared Error", value_abbrev="MSE", title_addition=title_addition, filepath=filepath, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "mae": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=mean_absolute_error, value_name="Mean Absolute Error", value_abbrev="MAE", title_addition=title_addition, filepath=filepath, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "medae": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=median_absolute_error, value_name="Median Absolute Error", value_abbrev="MedAE", title_addition=title_addition, filepath=filepath, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "mape": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=mean_absolute_percentage_error, value_name="Mean Absolute Percentage Error", value_abbrev="MAPE", title_addition=title_addition, filepath=filepath, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "maxe": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=max_error, value_name="Maximum Error", value_abbrev="ME", title_addition=title_addition, filepath=filepath, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "r2": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=r2_score, value_name="Coefficient of Determination (R-Squared)", value_abbrev="R^2", title_addition=title_addition, filepath=filepath, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "pearsonr": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=pearsonr, value_name="Pearson Correlation Coefficient", value_abbrev="R-VALUE", title_addition=title_addition, filepath=filepath, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "rm2": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=get_rm2, value_name="Modified Coefficient of Determination (R-Squared)", value_abbrev="RM^2", title_addition=title_addition, filepath=filepath, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "ci": dict(
            object=plot_regression, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=get_cindex, value_name="Concordance Index", value_abbrev="C-INDEX", title_addition=title_addition, filepath=filepath, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
    }
    reg_metrics = get_metrics_reg(y_true, y_pred, with_rm2, with_ci)
    
    if plot_method in plot_object_dict_kwargs and plot_method in reg_metrics:
        plot_object = plot_object_dict_kwargs[plot_method]["object"]
        plot_kwargs = plot_object_dict_kwargs[plot_method]["kwargs"]
        return plot_object(**plot_kwargs)
    raise KeyError(f"Dictionary Key: {plot_method}, is not recognized in plot_object_dict or reg_metrics.")

def get_plot_method_cls(plot_method, y_true, y_pred, transform=torch.sigmoid, threshold=0.5, return_ax=True):
    plot_object_dict_kwargs = {
        "f1": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=f1_score, value_name="F1 Score", value_abbrev="F1", transform=transform, threshold=threshold, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "precision": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=precision_score, value_name="Precision Score", value_abbrev="PS", transform=transform, threshold=threshold, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "recall": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=recall_score, value_name="Recall Score", value_abbrev="RS", transform=transform, threshold=threshold, single_val=False, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "accuracy": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=accuracy_score, value_name="Accuracy Score", value_abbrev="AS", transform=transform, threshold=threshold, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "mcc": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=matthews_corrcoef, value_name="Matthew's Correlation Coefficient", value_abbrev="MCC", transform=transform, threshold=threshold, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "rocauc": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=roc_auc_score, value_name="Area Under the Receiver Operating Characteristic Curve", value_abbrev="ROC_AUC", transform=transform, threshold=threshold, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
        "prauc": dict(
            object=plot_classification, kwargs=dict(
                y_true=y_true, y_pred=y_pred, value_obj=auc, value_name="Area Under the Receiver Operating Characteristic Curve", value_abbrev="AUC", transform=transform, threshold=threshold, single_val=True, fig_size=(10, 6), return_ax=return_ax
                )
            ),
    }
    reg_metrics = get_metrics_cls(y_true, y_pred, transform=transform, threshold=threshold)
    
    if plot_method in plot_object_dict_kwargs and plot_method in reg_metrics:
        plot_object = plot_object_dict_kwargs[plot_method]["object"]
        plot_kwargs = plot_object_dict_kwargs[plot_method]["kwargs"]
        return plot_object(**plot_kwargs)
    raise KeyError(f"Dictionary Key: {plot_method}, is not recognized in plot_object_dict or reg_metrics.")

# Plotting all metrics at once
def plot_regression_metrics(y_true, y_pred, title_addition=None, filepath=None, with_rm2=False, with_ci=False):
    reg_metrics = get_metrics_reg(y_true, y_pred, with_rm2, with_ci)

    # Determine the number of rows and columns for the subplot grid
    num_metrics = len(reg_metrics)
    num_cols = int(math.ceil(math.sqrt(num_metrics)))
    num_rows = int(math.ceil(num_metrics / num_cols))

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    if num_rows > 1:
        axes = axes.flatten()  # Flatten the axes array for easy indexing
    else:
        axes = [axes]  # Wrap in a list if only one row

    for idx, metric_name in enumerate(reg_metrics.keys()):
        kwargs = dict(
            plot_method=metric_name, y_true=y_true, y_pred=y_pred,
            title_addition=title_addition, filepath=filepath,
            with_rm2=with_rm2, with_ci=with_ci, return_ax=True
        )
        
        # Generate the plots and retrieve the axes
        ax1, ax2 = get_plot_method_reg(**kwargs)
        
        # Recreate lines from ax1 in the subplot axes
        for line in ax1.get_lines():
            axes[idx].plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())

        # Recreate collections (like scatter plots) from ax1 in the subplot axes
        for collection in ax1.collections:
            for path in collection.get_paths():
                patch = PathPatch(path, transform=axes[idx].transData, facecolor=collection.get_facecolor())
                axes[idx].add_patch(patch)

        # Set title, labels directly
        axes[idx].set_title(ax1.get_title())
        axes[idx].set_xlabel(ax1.get_xlabel())
        axes[idx].set_ylabel(ax1.get_ylabel())

        # Close the original plot to free up memory
        plt.close(ax1.figure)
        
        if ax2 is not None:
            # Create a secondary y-axis for the corresponding subplot
            secax = axes[idx].twinx()
    
            # Recreate lines from ax1 in the subplot axes
            for line in ax2.get_lines():
                secax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())

            # Recreate collections (like scatter plots) from ax1 in the subplot axes
            for collection in ax2.collections:
                for path in collection.get_paths():
                    patch = PathPatch(path, transform=secax.transData, facecolor=collection.get_facecolor())
                    secax.add_patch(patch)

            # Set title, labels directly
            secax.set_title(ax2.get_title())
            secax.set_xlabel(ax2.get_xlabel())
            secax.set_ylabel(ax2.get_ylabel())

            # Close the original plot to free up memory
            plt.close(ax2.figure)

    # Hide any unused axes in the figure
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

    # Save the figure if a filepath is provided
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')

    return fig, axes  # Return the figure and axes array

def generate_random_colors(n):
    palette = sns.color_palette("hsv", n)
    return palette.as_hex()

def plot_classification_metrics(y_true, y_pred, title_addition=None, filepath=None, fig_size=(30, 16), return_ax=True):
    json_path = "./symbol_to_idx.json"
    with open(json_path, "r") as f:
        symbol_to_idx = json.load(f)
    
    class_labels = list(symbol_to_idx.keys())
    n_classes = len(class_labels)

    # Binarize the output labels for multiclass
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()

    lines = []
    labels = []

    random_colors = generate_random_colors(n_classes)

    # Preparing ROC and PR curves
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

        color = random_colors[i]
        line, = plt.plot([], [], color=color, lw=2)
        lines.append(line)
        labels.append(f'{class_labels[i]} (ROC AUC = {roc_auc[i]:.2f}, PR AUC = {pr_auc[i]:.2f})')

    # Plotting ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    for i, color in zip(range(n_classes), random_colors):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=2)
        axes[1].plot(recall[i], precision[i], color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=20)
    axes[0].set_ylabel('True Positive Rate', fontsize=20)
    

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=20)
    axes[1].set_ylabel('Precision', fontsize=20)
    
    if title_addition:
        axes[0].set_title(title_addition + ' ROC Curves', fontsize=40)
        axes[1].set_title(title_addition + ' Precision-Recall Curves', fontsize=40)
    else:
        axes[0].set_title('ROC Curves', fontsize=40)
        axes[1].set_title('Precision-Recall Curves', fontsize=40)

    # Positioning the legend
    fig.legend(lines, labels, title="SELFIES STRING SECTIONS", loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.2))

    # Saving the figure with adjusted layout
    plt.show()
    if filepath:
        value_name = 'ROC and Precision-Recall Curves'
        new_file_path = filepath / value_name.replace(" ", "_")
        helper_io.create_folder(new_file_path)
        plt.savefig(new_file_path, bbox_inches='tight')
    plt.close()
        
    if return_ax:
        return fig, axes
    return

