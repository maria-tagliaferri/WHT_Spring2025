import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import constants
from utils.network import MyDataset
from utils.preprocessing import get_sensor_config
from model.Type3 import CNN_Alter_Block

# --- Desired exercise names for evaluation ---
DESIRED_EXERCISE_LIST = ['BulgSq', 'CMJDL', 'Run', 'SplitJump', 'SqDL', 'StepDnL', 'StepUpL', 'Walk']
# In the full exercise list (37 classes), these desired exercises are at the following indices:
# 'BulgSq' -> 0, 'CMJDL' -> 1, 'Run' -> 21, 'SplitJump' -> 25, 'SqDL' -> 27, 'StepDnL' -> 32, 'StepUpL' -> 34, 'Walk' -> 36.
mapping_dict = {0: 0, 1: 1, 21: 2, 25: 3, 27: 4, 32: 5, 34: 6, 36: 7}
# --- End desired exercise definitions ---

def test_loop(dataloader, model, loss_fn, device, num_desired):
    """
    Test loop that computes predictions.
    The model outputs 37-class predictions; we map these to 8 classes using mapping_dict.
    Ground-truth labels in the test data are assumed to already be in the 8-class format.
    Returns:
      - overall accuracy,
      - an 8x8 confusion matrix,
      - list of true labels, and list of mapped predicted labels.
    """
    total_samples = 0
    total_loss = 0
    all_true = []
    all_pred_mapped = []
    
    model.eval()
    with torch.no_grad():
        first_batch = True
        for X, y in dataloader:
            X = X.to(device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            else:
                y = y.float()
            y = y.to(device)
            
            # Model outputs 37-class predictions.
            pred = model(X)
            if first_batch:
                print("DEBUG: Prediction shape:", pred.shape)
                pred_np = pred.cpu().detach().numpy()
                print("DEBUG: Max predicted index (first batch):", np.max(np.argmax(pred_np, axis=1)))
                first_batch = False
                
            # Compute loss using ground truth (convert one-hot to indices).
            target = torch.argmax(y, dim=1)
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            batch_size = y.shape[0]
            total_samples += batch_size
            
            # Get raw predicted labels (from 37-class output).
            raw_pred = torch.argmax(pred, dim=1).cpu().numpy()
            # Map raw predictions to desired 8 classes.
            mapped_pred = []
            for p in raw_pred:
                if p in mapping_dict:
                    mapped_pred.append(mapping_dict[p])
                else:
                    # If prediction is not in mapping_dict, mark as invalid (-1)
                    mapped_pred.append(-1)
            
            true_labels = target.cpu().numpy()  # Ground truth should be in 0-7.
            
            all_pred_mapped.extend(mapped_pred)
            all_true.extend(true_labels.tolist())
    
    # Build confusion matrix for desired 8 classes.
    cm = np.zeros((num_desired, num_desired), dtype=int)
    correct = 0
    for t, p in zip(all_true, all_pred_mapped):
        if p < 0 or p >= num_desired:
            continue  # ignore invalid predictions
        cm[t, p] += 1
        if t == p:
            correct += 1
    accuracy = correct / total_samples
    avg_loss = total_loss / len(dataloader)
    return accuracy, cm, all_true, all_pred_mapped

def plot_confusion_matrix(cm, subject, classes=None, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots and saves the confusion matrix.
    """
    if normalize:
        cm = np.array(cm, dtype=np.float32)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums==0] = 1
        cm = cm / row_sums

    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"{title} - Subject {subject}")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    if classes is not None and len(classes)==cm.shape[0]:
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    else:
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs("conf_matrix", exist_ok=True)
    save_filename = f"conf_matrix/confusion_matrix_subject_{subject}.png"
    plt.savefig(save_filename)
    print(f"Confusion matrix for subject {subject} saved as {save_filename}")
    plt.close()

def main():
    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device for evaluation.")
    
    # Load test data.
    testing_file = "testing_processed/data_imu_10_both.pkl"
    with open(testing_file, "rb") as f:
        sample_list = pickle.load(f)
    print(f"Loaded {len(sample_list)} testing samples from {testing_file}")
    
    # Print structure of test data.
    exercise_codes = set()
    subject_codes = set()
    for sample in sample_list:
        ex_code = int(sample[0, constants.ID_EXERCISE_LABEL])
        sub_code = int(sample[0, constants.ID_SUBJECT_LABEL])
        exercise_codes.add(ex_code)
        subject_codes.add(sub_code)
    sorted_ex_codes = sorted(exercise_codes)
    print("Distinct exercise codes in test data:", sorted_ex_codes)
    print("Desired exercise names for evaluation:", DESIRED_EXERCISE_LIST)
    print("Distinct subject codes in test data:", sorted(subject_codes))
    
    # For evaluation, we use 8 classes.
    num_classes = 8
    
    # Get sensor configuration.
    config = get_sensor_config(n_sensors=10, sensor_pos_1="pelvis", sensor_pos_2="thigh_r", sensor_mod="both")
    
    # Build test dataset.
    # (Note: the test data should be labeled with 8-class one-hot labels corresponding to DESIRED_EXERCISE_LIST.)
    test_dataset = MyDataset(sample_list, constants.NORM_SAMPLE_LENGTH, num_classes=num_classes)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    conv_num_in = len(config.sensor_position) * len(config.sensor_modality) * constants.NUM_AX_PER_SENSOR
    
    all_results = {}
    
    # Get unique subject codes from test data.
    unique_subjects = sorted({int(sample[0, constants.ID_SUBJECT_LABEL]) for sample in sample_list})
    print("Unique subject codes in test data:", unique_subjects)
    
    for test_subject in range(19):
        print(f"\nEvaluating model {test_subject}")
        # Adjust hyperparameter lookup if training subjects were 1-indexed.
        hp_point = constants.TUNED_HP[test_subject + 1]  # mapping test subject code 0 -> subject 1, etc.
        s_num_out     = hp_point[constants.ID_NUM_OUT]
        s_kernel_size = hp_point[constants.ID_KERNEL_SIZE]
        s_stride      = hp_point[constants.ID_STRIDE]
        s_pool_size   = hp_point[constants.ID_POOL_SIZE]
        
        # Create the model.
        # Here, we load the 37-class model (trained on full list) and then map its predictions.
        model = CNN_Alter_Block(conv_num_in, s_num_out, s_kernel_size, s_stride, s_pool_size, num_classes=37)
        model.to(device)
        
        model_path = f"models/2025-02-18_12-43-13/trained_model_subject_{test_subject + 1}.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping subject {test_subject}.")
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Run test loop: map the 37-class outputs to the desired 8 using mapping_dict.
        acc, cm, all_true, all_pred = test_loop(test_loader, model, loss_fn, device, num_classes)
        print(f"Subject {test_subject} model test accuracy: {acc * 100:.2f}%")
        
        # Print per-exercise accuracies.
        print("Per-exercise accuracies:")
        for i in range(num_classes):
            total = np.sum(cm[i, :])
            correct = cm[i, i]
            if total > 0:
                ex_acc = correct / total * 100
                print(f"Exercise {i} ({DESIRED_EXERCISE_LIST[i]}): {ex_acc:.2f}% accuracy")
            else:
                print(f"Exercise {i} ({DESIRED_EXERCISE_LIST[i]}): No samples")
        
        all_results[test_subject] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "true": all_true,
            "pred": all_pred
        }
        
        # Plot and save the confusion matrix using DESIRED_EXERCISE_LIST as labels.
        plot_confusion_matrix(cm, test_subject, classes=DESIRED_EXERCISE_LIST, normalize=False,
                              title="Confusion Matrix (Exercises)")
    
    print("\nEvaluation complete. Summary of accuracies:")
    for subject, res in all_results.items():
        print(f"Subject {subject}: Accuracy = {res['accuracy'] * 100:.2f}%")

if __name__ == '__main__':
    main()