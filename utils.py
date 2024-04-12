import torch
import copy
import os
import torch.nn as nn
import torch.optim as optim
import importlib.util


def load_model(model_path: str, config) -> torch.nn.Module:
    """
    Load a PyTorch model from the specified file path.

    Parameters:
    - model_path (str): The file path of the model.

    Returns:
    - model (torch.nn.Module): The loaded PyTorch model.
    """
    # Loading model from file
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_class_name = config.get('model_class', 'Model')
    model = getattr(model_module, model_class_name)()

    return model
def train_kd(student_model, teacher_model, train_loader, T, alpha, epochs):
    """
        Train a student model using knowledge distillation.

        Parameters:
        - student_model (torch.nn.Module): The student model to be trained.
        - teacher_model (torch.nn.Module): The teacher model for knowledge distillation.
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        - T (float): Temperature parameter for distillation.
        - alpha (float): Weighting factor for combining hard target and distillation loss.
        - epochs (int): Number of training epochs.

        Returns:
        - trained_student_model (torch.nn.Module): The trained student model.
        """
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    criterion_distill = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            # Backpropagate the error and update the weights
            optimizer.zero_grad()
            student_logits = student_model(X_train)
            teacher_logits = teacher_model(X_train).detach()

            teacher_probs = torch.log_softmax(teacher_logits / T, dim=1)
            student_probs = torch.softmax(student_logits / T, dim=1)

            distillation_loss = criterion_distill(teacher_probs, student_probs)
            hard_target_loss = criterion(student_logits, y_train)

            loss = alpha * hard_target_loss + (1.0 - alpha) * T**2 * distillation_loss

            loss.backward()
            optimizer.step()

            # Print out some results every x times
            if b % 600 == 0:
                print(f'Epoch: {epoch}  Batch: {b}  Loss: {loss.item()}')

    return student_model

def print_model_shapes(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")

def get_weights(model):
    return [param.data for param in model.parameters()]

def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight



def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = (model, data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')






def average_weights(reconstructed_clients):
    num_clients = len(reconstructed_clients)
    num_weights = len(reconstructed_clients[0])

    summed_weights = []
    for i in range(num_weights):
        total_weight = sum(weights[i] for weights in reconstructed_clients)
        summed_weights.append(total_weight)

    # Calculate the average for each position
    averaged_weights = []
    for weight_sum in summed_weights:
        averaged_weight = weight_sum / num_clients
        averaged_weights.append(averaged_weight)

    print('Data has been averaged')

    return averaged_weights


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # size in MB
    os.remove('temp.p')
    return size


