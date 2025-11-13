import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import preprocess_input as pp_input

# Import from your existing code
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
                # Format: "001.Black_footed_Albatross"
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_id, class_name = parts
                    # Remove the numeric prefix for cleaner names
                    class_name = class_name.split('.', 1)[-1].replace('_', ' ')
                    class_names.append(class_name)
    else:
        print(f"Warning: classes.txt not found at {classes_file}")
        # Generate generic class names
        class_names = [f"Class_{i:03d}" for i in range(200)]
    
    return class_names


def evaluate_model_detailed(model, data_generator, class_names, output_dir):
    """
    Evaluate model and generate detailed per-class metrics
    
    Args:
        model: Trained Keras model
        data_generator: Data generator for test/validation data
        class_names: List of class names
        output_dir: Directory to save results
    """
    print("\n" + "="*80)
    print("EVALUATING MODEL - DETAILED PER-CLASS ANALYSIS")
    print("="*80)
    
    # Collect all predictions and labels
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
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Convert to class indices
    y_pred = np.argmax(all_predictions, axis=1)
    y_true = np.argmax(all_labels, axis=1)
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(y_pred == y_true)
    print(f"\n{'='*80}")
    print(f"OVERALL ACCURACY: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Generate classification report
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Create detailed per-class DataFrame
    per_class_results = []
    for i, class_name in enumerate(class_names):
        if class_name in report_dict:
            metrics = report_dict[class_name]
            per_class_results.append({
                'Class_ID': i,
                'Class_Name': class_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1-score'],
                'Support': int(metrics['support'])
            })
    
    df_results = pd.DataFrame(per_class_results)
    
    # Sort by F1-score to identify best and worst performing classes
    df_sorted = df_results.sort_values('F1_Score', ascending=False)
    
    # Print top 10 best performing classes
    print("\n" + "="*80)
    print("TOP 10 BEST PERFORMING CLASSES")
    print("="*80)
    for idx, row in df_sorted.head(10).iterrows():
        print(f"{row['Class_Name']:50s} | Precision: {row['Precision']:.4f} | "
              f"Recall: {row['Recall']:.4f} | F1: {row['F1_Score']:.4f} | "
              f"Support: {row['Support']}")
    
    # Print top 10 worst performing classes
    print("\n" + "="*80)
    print("TOP 10 WORST PERFORMING CLASSES")
    print("="*80)
    for idx, row in df_sorted.tail(10).iterrows():
        print(f"{row['Class_Name']:50s} | Precision: {row['Precision']:.4f} | "
              f"Recall: {row['Recall']:.4f} | F1: {row['F1_Score']:.4f} | "
              f"Support: {row['Support']}")
    
    # Print macro and weighted averages
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    print(f"Macro Avg    - Precision: {report_dict['macro avg']['precision']:.4f}, "
          f"Recall: {report_dict['macro avg']['recall']:.4f}, "
          f"F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg - Precision: {report_dict['weighted avg']['precision']:.4f}, "
          f"Recall: {report_dict['weighted avg']['recall']:.4f}, "
          f"F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print("="*80 + "\n")
    
    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'per_class_results.csv')
    df_sorted.to_csv(csv_path, index=False)
    print(f"✓ Per-class results saved to: {csv_path}")
    
    # Save full classification report
    report_text = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
        f.write(report_text)
    print(f"✓ Full classification report saved to: {report_path}")
    
    return df_results, overall_accuracy


def main():
    """Main evaluation function"""
    
    # Load configuration
    dataset_name = sys.argv[sys.argv.index('dataset') + 1] if 'dataset' in sys.argv else 'CUB200'
    param_dir = f"./configs/config_{dataset_name}.yaml"
    
    with open(param_dir, 'r') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    
    print('Loading configuration: \n', json.dumps(param, sort_keys=True, indent=3))
    
    # Extract parameters
    dataset = param['DATA']['dataset']
    rootdir = param['DATA']['rootdir']
    image_size = param['DATA']['image_size']
    batch_size = param['MODEL']['batch_size']
    model_name = param['MODEL']['model_name']
    backbone = param['MODEL']['backbone']
    
    # Get model checkpoint path
    if 'checkpoint_path' in sys.argv:
        checkpoint_path = sys.argv[sys.argv.index('checkpoint_path') + 1]
    else:
        checkpoint_path = param['MODEL']['checkpoint_path']
        # Find the best model in the directory
        if os.path.isdir(checkpoint_path):
            model_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.h5')]
            if model_files:
                # Sort by validation accuracy (assuming format includes valAcc)
                model_files.sort(reverse=True)
                checkpoint_path = os.path.join(checkpoint_path, model_files[0])
                print(f"\n✓ Found model: {checkpoint_path}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at: {checkpoint_path}")
        print("Please provide the path using: python evaluate_per_class.py dataset CUB200 checkpoint_path /path/to/model.h5")
        sys.exit(1)
    
    # Setup directories
    dataset_dir = rootdir
    test_data_dir = f'{dataset_dir}/test/'
    if not os.path.isdir(test_data_dir):
        test_data_dir = f'{dataset_dir}/val/'
    
    output_dir = './EvaluationResults/' + model_name
    
    # Load class names
    class_names = load_cub_class_names(dataset_dir)
    nb_classes = len(class_names)
    print(f'\nDataset: {dataset}')
    print(f'Number of classes: {nb_classes}')
    print(f'Test data directory: {test_data_dir}\n')
    
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(param['HARDWARE']['gpu_id'])
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=param['HARDWARE']['gpu_utilisation']
    )
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    # Build model architecture
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
    
    # Initialize model
    outputs = model(model.base_model.input)
    model = Model(inputs=model.input, outputs=outputs)
    
    # Load weights
    print(f"Loading model weights from: {checkpoint_path}")
    model.load_weights(checkpoint_path)
    print("✓ Model loaded successfully\n")
    
    # Create data generator for test set
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
    
    # Evaluate model
    df_results, overall_accuracy = evaluate_model_detailed(
        model, 
        test_dg, 
        class_names, 
        output_dir
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()