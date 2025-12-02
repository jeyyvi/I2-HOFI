import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import preprocess_input as pp_input

from dataset_info import datasetInfo
from datagen import DirectoryDataGenerator
from models import construct_model


def load_cub_class_names(data_dir):
    """Load class names from CUB-200 dataset"""
    classes_file = os.path.join(data_dir, "classes.txt")
    class_names = []
    
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_id, class_name = parts
                    class_name = class_name.split('.', 1)[-1].replace('_', ' ')
                    class_names.append(class_name)
    else:
        print(f"Warning: classes.txt not found at {classes_file}")
        class_names = [f"Class_{i:03d}" for i in range(200)]
    
    return class_names


def calculate_per_class_accuracy(y_true, y_pred, num_classes):
    """Calculate accuracy for each class individually"""
    per_class_acc = {}
    
    for class_id in range(num_classes):
        # Get indices where true label is this class
        class_mask = (y_true == class_id)
        
        if np.sum(class_mask) > 0:  # If this class exists in the data
            class_true = y_true[class_mask]
            class_pred = y_pred[class_mask]
            accuracy = accuracy_score(class_true, class_pred)
            per_class_acc[class_id] = accuracy
        else:
            per_class_acc[class_id] = None
    
    return per_class_acc


def evaluate_model_detailed(model, data_generator, class_names, output_dir):
    """Comprehensive evaluation with all metrics"""
    
    print("\n" + "="*80)
    print("EVALUATING MODEL - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    
    print("\nGenerating predictions...")
    for i, (batch_x, batch_y) in enumerate(data_generator):
        if i >= len(data_generator):
            break
        
        predictions = model.predict(batch_x, verbose=0)
        all_predictions.append(predictions)
        all_labels.append(batch_y)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data_generator)} batches")
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    y_pred = np.argmax(all_predictions, axis=1)
    y_true = np.argmax(all_labels, axis=1)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Per-class accuracy
    per_class_accuracy = calculate_per_class_accuracy(y_true, y_pred, len(class_names))
    
    # Classification report
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Build comprehensive per-class results
    per_class_results = []
    for i, class_name in enumerate(class_names):
        if class_name in report_dict:
            metrics = report_dict[class_name]
            per_class_results.append({
                'Class_ID': i,
                'Class_Name': class_name,
                'Accuracy': per_class_accuracy.get(i, 0.0) if per_class_accuracy.get(i) is not None else 0.0,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    df_results = pd.DataFrame(per_class_results)
    
    # Sort by accuracy
    df_sorted = df_results.sort_values('Accuracy', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("TOP 10 BEST PERFORMING CLASSES (by Accuracy)")
    print("="*80)
    for idx, row in df_sorted.head(10).iterrows():
        print(f"{row['Class_Name']:45s} | Acc: {row['Accuracy']:.4f} | "
              f"Prec: {row['Precision']:.4f} | Rec: {row['Recall']:.4f} | "
              f"F1: {row['F1_Score']:.4f} | N={row['Support']}")
    
    print("\n" + "="*80)
    print("TOP 10 WORST PERFORMING CLASSES (by Accuracy)")
    print("="*80)
    for idx, row in df_sorted.tail(10).iterrows():
        print(f"{row['Class_Name']:45s} | Acc: {row['Accuracy']:.4f} | "
              f"Prec: {row['Precision']:.4f} | Rec: {row['Recall']:.4f} | "
              f"F1: {row['F1_Score']:.4f} | N={row['Support']}")
    
    # Aggregate metrics
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    print(f"Overall Accuracy:  {overall_accuracy:.4f}")
    print(f"Macro Avg    - Precision: {report_dict['macro avg']['precision']:.4f}, "
          f"Recall: {report_dict['macro avg']['recall']:.4f}, "
          f"F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg - Precision: {report_dict['weighted avg']['precision']:.4f}, "
          f"Recall: {report_dict['weighted avg']['recall']:.4f}, "
          f"F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"Mean Per-Class Accuracy: {df_results['Accuracy'].mean():.4f}")
    print("="*80 + "\n")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-class results CSV
    csv_path = os.path.join(output_dir, 'per_class_results.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"✓ Per-class results saved to: {csv_path}")
    
    # Save detailed classification report
    report_text = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Mean Per-Class Accuracy: {df_results['Accuracy'].mean():.4f}\n\n")
        f.write("-"*80 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n\n")
        f.write(report_text)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("AGGREGATE METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Macro Average - Precision: {report_dict['macro avg']['precision']:.4f}, "
                f"Recall: {report_dict['macro avg']['recall']:.4f}, "
                f"F1: {report_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Average - Precision: {report_dict['weighted avg']['precision']:.4f}, "
                f"Recall: {report_dict['weighted avg']['recall']:.4f}, "
                f"F1: {report_dict['weighted avg']['f1-score']:.4f}\n")
    
    print(f"✓ Full classification report saved to: {report_path}")
    
    # Generate plots
    create_evaluation_plots(df_results, output_dir)
    
    return df_results, overall_accuracy


def create_evaluation_plots(df_results, output_dir):
    """Create visualization plots for evaluation metrics"""
    
    print("\nGenerating visualization plots...")
    
    # 1. Metrics distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    colors = ['skyblue', 'lightgreen', 'salmon', 'wheat']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        row = idx // 3
        col = idx % 3
        
        axes[row, col].hist(df_results[metric], bins=30, color=color, edgecolor='black', alpha=0.7)
        axes[row, col].set_title(f'{metric.replace("_", " ")} Distribution', fontsize=14, fontweight='bold')
        axes[row, col].set_xlabel(metric.replace("_", " "), fontsize=12)
        axes[row, col].set_ylabel('Number of Classes', fontsize=12)
        axes[row, col].axvline(df_results[metric].mean(), color='red', linestyle='--', 
                               linewidth=2, label=f'Mean: {df_results[metric].mean():.3f}')
        axes[row, col].legend()
        axes[row, col].grid(alpha=0.3)
    
    # Support distribution
    axes[1, 1].hist(df_results['Support'], bins=30, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Support Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 1].set_ylabel('Number of Classes', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    # Hide empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'metrics_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics distributions saved to: {dist_path}")
    
    # 2. Top/Bottom classes comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Top 15 classes
    top_15 = df_results.nlargest(15, 'Accuracy')
    axes[0].barh(range(len(top_15)), top_15['Accuracy'], color='green', alpha=0.7)
    axes[0].set_yticks(range(len(top_15)))
    axes[0].set_yticklabels(top_15['Class_Name'], fontsize=10)
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Top 15 Best Performing Classes', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Bottom 15 classes
    bottom_15 = df_results.nsmallest(15, 'Accuracy')
    axes[1].barh(range(len(bottom_15)), bottom_15['Accuracy'], color='red', alpha=0.7)
    axes[1].set_yticks(range(len(bottom_15)))
    axes[1].set_yticklabels(bottom_15['Class_Name'], fontsize=10)
    axes[1].set_xlabel('Accuracy', fontsize=12)
    axes[1].set_title('Bottom 15 Worst Performing Classes', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'top_bottom_classes.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Top/bottom classes comparison saved to: {comparison_path}")
    
    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_results[['Accuracy', 'Precision', 'Recall', 'F1_Score']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'metrics_correlation.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics correlation saved to: {corr_path}")


def main():
    """Main evaluation function"""
    
    dataset_name = sys.argv[sys.argv.index('dataset') + 1] if 'dataset' in sys.argv else 'CUB200'
    param_dir = f"./configs/config_{dataset_name}.yaml"
    
    with open(param_dir, 'r') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    
    print('Loading configuration: \n', json.dumps(param, sort_keys=True, indent=3))
    
    dataset = param['DATA']['dataset']
    rootdir = param['DATA']['rootdir']
    image_size = param['DATA']['image_size']
    batch_size = param['MODEL']['batch_size']
    model_name = param['MODEL']['model_name']
    backbone = param['MODEL']['backbone']
    
    if 'checkpoint_path' in sys.argv:
        checkpoint_path = sys.argv[sys.argv.index('checkpoint_path') + 1]
    else:
        checkpoint_path = param['MODEL']['checkpoint_path']
        if os.path.isdir(checkpoint_path):
            model_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.h5')]
            if model_files:
                model_files.sort(reverse=True)
                checkpoint_path = os.path.join(checkpoint_path, model_files[0])
                print(f"\n✓ Found model: {checkpoint_path}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at: {checkpoint_path}")
        sys.exit(1)
    
    dataset_dir = rootdir
    test_data_dir = f'{dataset_dir}/test/'
    if not os.path.isdir(test_data_dir):
        test_data_dir = f'{dataset_dir}/val/'
    
    output_dir = '/content/drive/MyDrive/I2HOFI_Models2/EvaluationResults/' + model_name
    
    class_names = load_cub_class_names(dataset_dir)
    nb_classes = len(class_names)
    print(f'\nDataset: {dataset}')
    print(f'Number of classes: {nb_classes}')
    print(f'Test data directory: {test_data_dir}\n')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param['HARDWARE']['gpu_id'])
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=param['HARDWARE']['gpu_utilisation']
    )
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    print("Building model architecture...")
    model = construct_model(
        name=model_name,
        pool_size=param['HYPERPARAMETERS']['pool_size'],
        ROIS_resolution=param['HYPERPARAMETERS']['ROIS_resolution'],
        ROIS_grid_size=param['HYPERPARAMETERS']['ROIS_grid_size'],
        minSize=param['HYPERPARAMETERS']['minSize'],
        alpha=param['HYPERPARAMETERS']['alpha'],
        nb_classes=nb_classes,
        batch_size=batch_size,
        input_sh=image_size,
        gcn_outfeat_dim=param['HYPERPARAMETERS']['gcn_outfeat_dim'],
        gat_outfeat_dim=param['HYPERPARAMETERS']['gat_outfeat_dim'],
        dropout_rate=param['HYPERPARAMETERS']['dropout_rate'],
        l2_reg=param['HYPERPARAMETERS']['l2_reg'],
        attn_heads=param['HYPERPARAMETERS']['attn_heads'],
        appnp_activation=param['HYPERPARAMETERS']['appnp_activation'],
        gat_activation=param['HYPERPARAMETERS']['gat_activation'],
        concat_heads=param['HYPERPARAMETERS']['concat_heads'],
        backbone=backbone,
        freeze_backbone=param['MODEL']['freeze_backbone'],
        gnn1_layr=param['MODEL']['gnn1_layr'],
        gnn2_layr=param['MODEL']['gnn2_layr'],
        track_feat=False,
    )
    
    outputs = model(model.base_model.input)
    model = Model(inputs=model.input, outputs=outputs)
    
    print(f"Loading model weights from: {checkpoint_path}")
    model.load_weights(checkpoint_path)
    print("✓ Model loaded successfully\n")
    
    print("Setting up data generator...")
    test_dg = DirectoryDataGenerator(
        base_directories=[test_data_dir],
        augmentor=None,
        target_sizes=image_size[:2],
        preprocessors=pp_input,
        batch_size=batch_size,
        shuffle=False,
        channel_last=True,
        verbose=1,
        hasROIS=False
    )
    
    df_results, overall_accuracy = evaluate_model_detailed(
        model, test_dg, class_names, output_dir
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()