# Authors: Rohan Subramanian, Sahil Rane and Forrest Bicker
# Training loop function adapted from Sebastian Raschka 
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb

# Import statements
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def compute_accuracy(model, data_loader, device):
    """Compute the overall accuracy of the model on the given dataset."""
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def compute_class_accuracy(model, data_loader, device):
    """Compute the accuracy for each class in the dataset."""
    model.eval()
    
    with torch.no_grad():
        class_correct = dict()
        class_totals = dict()
    
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
    
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
    
            for target, prediction in zip(targets, predicted_labels):
                if target.item() not in class_correct:
                    class_correct[target.item()] = 0
                    class_totals[target.item()] = 0
                if prediction == target:
                    class_correct[target.item()] += 1
                class_totals[target.item()] += 1
    
        class_accuracies = {cls: (class_correct[cls] / class_totals[cls]) * 100 
                            for cls in class_correct}

    return class_accuracies

def compute_epoch_loss(model, data_loader, device):
    """Compute the average loss of the model over all batches in the dataset."""
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_roc_auc(model, data_loader, device, num_classes):
    """Compute the ROC-AUC score for a multi-class classification model."""
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    prediction_scores = []

    with torch.no_grad():  # No need to track gradients for validation
        for features, targets in data_loader:
            features = features.to(device)

            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)  # Use softmax for multi-class classification

            true_labels.extend(targets.cpu().numpy())  # Move to CPU and convert to numpy
            prediction_scores.extend(probabilities.cpu().numpy())

    true_labels = label_binarize(true_labels, classes=range(num_classes))
    roc_auc = roc_auc_score(true_labels, prediction_scores, multi_class='ovr')
    return roc_auc

def compute_confusion_matrix(model, data_loader, device):
    """Compute and return the confusion matrix for the model's predictions."""
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # No need to track gradients for validation
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            _, predicted_labels = torch.max(outputs, 1)  # Get the index of the max log-probability

            all_predictions.extend(predicted_labels.cpu().numpy())  # Move to CPU and convert to numpy
            all_targets.extend(targets.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    return cm

def train_classifier(
        model, num_epochs, train_loader,
        valid_loader, test_loader, optimizer,
        device, logging_interval=50,
        best_model_save_path=None,
        scheduler=None,
        skip_train_acc=False,
        scheduler_on='valid_acc'):
    """Train a classifier and return the training, validation accuracy, and loss lists."""

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    best_valid_acc, best_epoch = -float('inf'), 0

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():  # save memory during inference
            if not skip_train_acc:
                train_acc = compute_accuracy(model, train_loader, device=device).item()
            else:
                train_acc = float('nan')
            valid_acc = compute_accuracy(model, valid_loader, device=device).item()
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc, best_epoch = valid_acc, epoch+1
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}% '
                  f'| Best Validation '
                  f'(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%')

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError('Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list