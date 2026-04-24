"""
Self-Pruning Neural Network with Dynamic Pruning During Training

This implementation demonstrates learnable pruning gates applied during training.
The network learns to prune its own weights by optimizing gate parameters alongside weights.

Author: ML Engineer
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os


# PART 1: CUSTOM PRUNABLE LINEAR LAYER

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable pruning gates.
    
    Key Concept:
    - Each weight has a corresponding learnable gate parameter
    - Gates are learned during training to prune unimportant weights
    - The pruned weight is computed as: pruned_weight = weight * sigmoid(gate_scores)
    
    Why this works:
    - sigmoid(x) produces values in (0, 1), acting as a soft masking function
    - During backprop, gradients flow through BOTH weight and gate_scores
    - Sparsity regularization encourages gates toward 0 (pruning)
    - Hard pruning can be applied after training by thresholding gates
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
        """
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameter (trainable)
        # Standard weights that get modified by gates
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Gate scores parameter (trainable, SAME shape as weights)
        # These are passed through sigmoid to create soft masks
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias (optional standard parameter)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights and gates using standard initialization."""
        # Initialize weights using Kaiming uniform (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        # Initialize gate_scores to be close to 0
        # This means sigmoid(gate_scores) ≈ 0.5 initially (half active)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.1)
        
        # Initialize bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass with learnable pruning gates.
        
        Process:
        1. Compute gates: sigmoid(gate_scores) → values in (0, 1)
        2. Apply gates to weights: pruned_weight = weight * gates
        3. Standard linear transformation: output = x @ pruned_weight.T + bias
        
        Key: Gradients flow through both weight and gate_scores parameters.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Compute gates using sigmoid (output range: 0 to 1)
        # sigmoid(high value) ≈ 1 (weight active)
        # sigmoid(low value) ≈ 0 (weight pruned)
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        # This creates soft pruning: weights are multiplied by gate values
        pruned_weight = self.weight * gates
        
        # Perform standard linear transformation with pruned weights
        # Uses optimized PyTorch function
        return torch.nn.functional.linear(x, pruned_weight, self.bias)
    
    def get_sparsity_loss(self):
        """
        Compute sparsity loss as L1 regularization on gates.
        
        Why L1 on gates promotes sparsity:
        - Sigmoid outputs are in range (0, 1)
        - Summing gate values creates an L1 penalty
        - L1 is known to promote sparsity (compared to L2)
        - Low gate value = weight effectively pruned (multiplied by small number)
        - During training, sparsity loss encourages gates toward 0
        
        Why sigmoid is used:
        - Sigmoid is differentiable everywhere (safe for backprop)
        - Output range (0, 1) is intuitive for gates (probability of keeping)
        - Smooth transition: easier to learn than hard thresholding
        
        Returns:
            Scalar tensor representing sum of all gates in this layer
        """
        gates = torch.sigmoid(self.gate_scores)
        # Sum all gate values across all weights
        # Higher λ multiplier will push these values lower
        return gates.sum()


# PART 2: MODEL ARCHITECTURE

class PrunableNetwork(nn.Module):
    """
    Feedforward neural network using PrunableLinear layers.
    
    Architecture:
    - Input (3072 features from CIFAR-10 flattened) 
    - PrunableLinear → ReLU
    - PrunableLinear → ReLU
    - PrunableLinear → Output (10 classes for CIFAR-10)
    
    All weights in all linear layers can be pruned during training.
    """
    
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10):
        """
        Args:
            input_size: Size of flattened input (3 * 32 * 32 for CIFAR-10)
            hidden_size: Number of hidden units in hidden layers
            num_classes: Number of output classes (10 for CIFAR-10)
        """
        super(PrunableNetwork, self).__init__()
        
        # First hidden layer with prunable weights
        self.layer1 = PrunableLinear(input_size, hidden_size)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second hidden layer with prunable weights
        self.layer2 = PrunableLinear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Output layer with prunable weights
        self.layer3 = PrunableLinear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32) from CIFAR-10
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Flatten input: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)
        
        # First layer + activation
        x = self.relu1(self.layer1(x))
        
        # Second layer + activation
        x = self.relu2(self.layer2(x))
        
        # Output layer (no activation, logits for cross-entropy)
        x = self.layer3(x)
        
        return x
    
    def get_total_sparsity_loss(self):
        """
        Compute total sparsity loss across all prunable layers.
        
        Iterates through all modules in the network and sums sparsity loss
        from all PrunableLinear layers.
        
        Returns:
            Scalar tensor representing total sparsity loss across network
        """
        sparsity_loss = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                sparsity_loss += module.get_sparsity_loss()
        return sparsity_loss


# PART 3: LOSS FUNCTION

def compute_total_loss(outputs, targets, model, lambda_sparsity):
    """
    Compute combined loss function for training.
    
    Total Loss = CrossEntropyLoss + λ * SparsityLoss
    
    Components:
    - CrossEntropyLoss: Measures classification error
    - SparsityLoss: Sum of all gate values (L1 regularization)
    - λ (lambda_sparsity): Hyperparameter controlling trade-off
    
    Trade-off explanation:
    - Higher λ: Stronger pruning regularization, more sparse network, lower accuracy
    - Lower λ: Weaker pruning regularization, denser network, higher accuracy
    - Optimal λ: Depends on application (accuracy vs efficiency trade-off)
    
    Why this formulation works:
    1. CrossEntropyLoss ensures the model learns to classify correctly
    2. SparsityLoss (L1 on gates) encourages zeros/small values in gate_scores
    3. When gate_scores are small, sigmoid(gate_scores) ≈ 0.5, but during optimization
       the sparsity term pushes gates lower, effectively pruning weights
    4. The λ parameter allows us to control how aggressively we prune
    
    Args:
        outputs: Model predictions (logits), shape (batch_size, num_classes)
        targets: Ground truth labels, shape (batch_size,)
        model: PrunableNetwork instance
        lambda_sparsity: Regularization coefficient (λ)
    
    Returns:
        total_loss: Combined loss (used for backprop)
        ce_loss: Classification loss only (for monitoring)
        sparsity_loss: Sparsity regularization loss (for monitoring)
    """
    # Classification loss: cross-entropy between predictions and targets
    ce_loss = nn.functional.cross_entropy(outputs, targets)
    
    # Sparsity regularization: L1 penalty on gate values
    sparsity_loss = model.get_total_sparsity_loss()
    
    # Combined loss: weighted sum
    total_loss = ce_loss + lambda_sparsity * sparsity_loss
    
    return total_loss, ce_loss, sparsity_loss


# PART 4: SPARSITY METRIC

def calculate_sparsity(model, threshold=1e-2):
    """
    Calculate the sparsity level of the network.
    
    Definition:
    Sparsity (%) = (Number of pruned weights / Total weights) * 100
    where "pruned weight" is defined as gate < threshold
    
    Interpretation:
    - Higher sparsity = more weights have been pruned (low gate values)
    - Lower sparsity = most weights are active (high gate values)
    - threshold=1e-2 means gates with value < 0.01 are considered pruned
    
    Rationale:
    - We use gate value (not weight value) because gates directly control pruning
    - A low gate means the weight is multiplied by a small value (effectively inactive)
    - Threshold of 1e-2 is arbitrary but reasonable for sigmoid outputs
    
    Args:
        model: PrunableNetwork instance
        threshold: Gate value threshold for considering a weight pruned
    
    Returns:
        sparsity_percent: Percentage of pruned weights (0-100)
    """
    total_params = 0
    pruned_params = 0
    
    # Iterate through all prunable layers
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # Compute gate values: sigmoid(gate_scores) ∈ (0, 1)
            gates = torch.sigmoid(module.gate_scores)
            
            # Count total parameters
            total_params += gates.numel()
            
            # Count parameters where gate < threshold (considered pruned)
            pruned_params += (gates < threshold).sum().item()
    
    # Avoid division by zero
    if total_params == 0:
        return 0.0
    
    # Calculate sparsity percentage
    sparsity_percent = (pruned_params / total_params) * 100
    return sparsity_percent


# PART 5: DATA LOADING

def load_cifar10_data(batch_size=128, num_workers=2):
    """
    Load CIFAR-10 dataset with standard preprocessing.
    
    CIFAR-10:
    - 60,000 images (32x32 pixels)
    - 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    - 50,000 training images
    - 10,000 test images
    
    Preprocessing:
    - Training: Random crop, random flip, normalization
    - Testing: Only normalization (no augmentation)
    - Normalization: Mean and std of ImageNet (standard practice)
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of parallel workers for data loading
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    # Transformations for training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Augmentation: random crop with padding
        transforms.RandomHorizontalFlip(),      # Augmentation: random flip
        transforms.ToTensor(),                  # Convert to tensor
        # Normalization: mean and std of ImageNet
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Transformations for test set (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training set
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    # Load test set
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


# PART 6: TRAINING LOOP

def train_epoch(model, train_loader, optimizer, device, lambda_sparsity):
    """
    Train the model for one epoch.
    
    Process:
    1. Set model to training mode
    2. For each batch:
       a. Forward pass through network
       b. Compute loss (classification + sparsity regularization)
       c. Backward pass (compute gradients)
       d. Optimizer step (update weights and gates)
    3. Track and return loss metrics
    
    Args:
        model: PrunableNetwork instance
        train_loader: DataLoader for training set
        optimizer: Optimizer instance (Adam)
        device: Device to run on (CPU/GPU)
        lambda_sparsity: Regularization coefficient
    
    Returns:
        avg_loss: Average total loss across all batches
        avg_ce_loss: Average classification loss across all batches
        avg_sparsity_loss: Average sparsity loss across all batches
    """
    model.train()  # Set to training mode (enables dropout, batch norm updates, etc.)
    
    total_loss = 0
    total_ce_loss = 0
    total_sparsity_loss = 0
    
    # Iterate through batches
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device (GPU if available)
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Compute loss
        loss, ce_loss, sparsity_loss = compute_total_loss(
            outputs, targets, model, lambda_sparsity)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Optimizer step: update weights and gates
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sparsity_loss += sparsity_loss.item()
    
    # Average losses across all batches
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_sparsity_loss = total_sparsity_loss / num_batches
    
    return avg_loss, avg_ce_loss, avg_sparsity_loss


# PART 7: EVALUATION FUNCTION

def evaluate(model, test_loader, device):
    """
    Evaluate model accuracy on test set.
    
    Process:
    1. Set model to evaluation mode (disable dropout, use running batch norm stats)
    2. Disable gradient computation (for efficiency)
    3. For each batch:
       a. Forward pass
       b. Get predicted class
       c. Compare with ground truth
    4. Compute overall accuracy
    
    Args:
        model: PrunableNetwork instance
        test_loader: DataLoader for test set
        device: Device to run on
    
    Returns:
        accuracy: Percentage of correct predictions (0-100)
    """
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, targets in test_loader:
            # Move to device
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            
            # Count correct predictions
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Compute accuracy percentage
    accuracy = 100 * correct / total
    return accuracy


# PART 8: MAIN TRAINING PIPELINE

def train_model(lambda_sparsity, num_epochs=20, batch_size=128, learning_rate=0.001):
    """
    Train a prunable network with specified sparsity regularization.
    
    This is the main training loop that:
    1. Initializes model and optimizer
    2. Trains for multiple epochs
    3. Evaluates on test set after each epoch
    4. Tracks metrics and sparsity
    
    Args:
        lambda_sparsity: Sparsity regularization coefficient (λ)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        model: Trained PrunableNetwork
        history: Dictionary containing training history
        test_accuracy: Final test accuracy
        sparsity: Final sparsity percentage
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training with λ = {lambda_sparsity}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    # Load data
    train_loader, test_loader = load_cifar10_data(batch_size=batch_size)
    
    # Create model
    model = PrunableNetwork(input_size=3072, hidden_size=512, num_classes=10)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Optimizer: Adam is good for this task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler: decay LR by 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'ce_loss': [],
        'sparsity_loss': [],
        'test_accuracy': [],
        'sparsity_percent': []
    }
    
    # Training loop
    print(f"\n{'Epoch':<8} {'Total Loss':<15} {'CE Loss':<15} {'Sparsity Loss':<15} {'Test Acc %':<15} {'Sparsity %':<15}")
    print("-" * 90)
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, ce_loss, sparsity_loss = train_epoch(
            model, train_loader, optimizer, device, lambda_sparsity
        )
        
        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)
        
        # Calculate sparsity metric
        sparsity = calculate_sparsity(model)
        
        # Store history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['ce_loss'].append(ce_loss)
        history['sparsity_loss'].append(sparsity_loss)
        history['test_accuracy'].append(test_acc)
        history['sparsity_percent'].append(sparsity)
        
        # Print progress
        print(f"{epoch+1:<8} {train_loss:<15.4f} {ce_loss:<15.4f} {sparsity_loss:<15.4f} {test_acc:<15.2f} {sparsity:<15.2f}")
        
        # Step learning rate scheduler
        scheduler.step()
    
    print(f"{'='*70}")
    
    return model, history, test_acc, sparsity


# PART 9: VISUALIZATION FUNCTIONS

def plot_gate_distributions(models_dict):
    """
    Plot histogram of gate values for different lambda values.
    
    Visualization shows:
    - X-axis: Gate value (0 to 1, from sigmoid output)
    - Y-axis: Frequency (number of gates with that value)
    
    Expected patterns:
    - Low λ: Smoother distribution (less pruning)
    - High λ: Spike near 0 (aggressive pruning)
    
    This histogram visually shows the sparsity trade-off.
    
    Args:
        models_dict: Dictionary mapping lambda values to trained models
    """
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5*len(models_dict), 4))
    
    # Handle single subplot case
    if len(models_dict) == 1:
        axes = [axes]
    
    # Plot for each lambda value
    for idx, (lambda_val, model) in enumerate(models_dict.items()):
        all_gates = []
        
        # Collect all gate values from all layers
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
                all_gates.extend(gates.flatten())
        
        all_gates = np.array(all_gates)
        
        # Plot histogram
        axes[idx].hist(all_gates, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'Gate Distribution\n(λ = {lambda_val})', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Gate Value (sigmoid output)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_gate = np.mean(all_gates)
        std_gate = np.std(all_gates)
        axes[idx].text(0.98, 0.97, f'μ={mean_gate:.3f}\nσ={std_gate:.3f}',
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('gate_distributions.png', dpi=150, bbox_inches='tight')
    print("Gate distributions saved to 'gate_distributions.png'")
    plt.close()


def plot_training_metrics(histories_dict):
    """
    Plot training metrics across epochs for different lambda values.
    
    Four subplots showing:
    1. Classification loss (CE) vs epoch
    2. Sparsity loss vs epoch
    3. Test accuracy vs epoch
    4. Sparsity percentage vs epoch
    
    This visualization shows the training dynamics and accuracy-sparsity trade-off.
    
    Args:
        histories_dict: Dictionary mapping lambda values to training histories
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot for each lambda value
    for lambda_val, history in histories_dict.items():
        axes[0, 0].plot(history['epoch'], history['ce_loss'], 
                       label=f'λ = {lambda_val}', marker='o', markersize=3)
        axes[0, 1].plot(history['epoch'], history['sparsity_loss'], 
                       label=f'λ = {lambda_val}', marker='o', markersize=3)
        axes[1, 0].plot(history['epoch'], history['test_accuracy'], 
                       label=f'λ = {lambda_val}', marker='o', markersize=3)
        axes[1, 1].plot(history['epoch'], history['sparsity_percent'], 
                       label=f'λ = {lambda_val}', marker='o', markersize=3)
    
    # Configure subplots
    axes[0, 0].set_title('Classification Loss (CE)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('CE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_title('Sparsity Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Sparsity Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_title('Test Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_title('Network Sparsity Level', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sparsity (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Training metrics saved to 'training_metrics.png'")
    plt.close()


# PART 10: MAIN EXECUTION

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SELF-PRUNING NEURAL NETWORK - DYNAMIC PRUNING DURING TRAINING")
    print("="*70)
    print("\nThis script demonstrates learnable pruning using gate parameters.")
    print("The network learns to prune its own weights during training.\n")
    
    # HYPERPARAMETERS
    
    lambda_values = [1e-5, 1e-4, 1e-3]  # Different sparsity regularization strengths
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    
    print("Configuration:")
    print(f"  Lambda values: {lambda_values}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}\n")
    
    # TRAIN MODELS WITH DIFFERENT LAMBDA VALUES
    
    results = []
    models_dict = {}
    histories_dict = {}
    
    for lambda_val in lambda_values:
        model, history, test_acc, sparsity = train_model(
            lambda_sparsity=lambda_val,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        results.append({
            'lambda': lambda_val,
            'accuracy': test_acc,
            'sparsity': sparsity
        })
        
        models_dict[lambda_val] = model
        histories_dict[lambda_val] = history
    
    # PRINT RESULTS TABLE
    
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS TABLE")
    print("="*70)
    print(f"{'Lambda':<15} {'Accuracy (%)':<20} {'Sparsity (%)':<20}")
    print("-" * 70)
    for result in results:
        print(f"{result['lambda']:<15.1e} {result['accuracy']:<20.2f} {result['sparsity']:<20.2f}")
    print("="*70 + "\n")
    
    # ANALYSIS
    
    print("TRADE-OFF ANALYSIS:")
    print("-" * 70)
    
    # Find best accuracy and best sparsity
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_sparse = max(results, key=lambda x: x['sparsity'])
    
    print(f"Highest Accuracy: λ = {best_acc['lambda']:.1e}, Acc = {best_acc['accuracy']:.2f}%")
    print(f"Highest Sparsity: λ = {best_sparse['lambda']:.1e}, Sparsity = {best_sparse['sparsity']:.2f}%")
    
    # Calculate accuracy drop for sparsity gain
    acc_range = best_acc['accuracy'] - min(r['accuracy'] for r in results)
    sparse_range = best_sparse['sparsity'] - min(r['sparsity'] for r in results)
    
    print(f"\nAccuracy range: {acc_range:.2f}% (lowest λ to highest λ)")
    print(f"Sparsity range: {sparse_range:.2f}% (lowest λ to highest λ)")
    print("\nObservation: Higher λ increases sparsity at the cost of accuracy.")
    print("This demonstrates the trade-off between model efficiency and performance.\n")
    
    # GENERATE VISUALIZATIONS
    
    print("Generating visualizations...")
    plot_gate_distributions(models_dict)
    plot_training_metrics(histories_dict)
    
    # SAVE RESULTS
    
    print("\nSaving results to file...")
    with open('results.txt', 'w') as f:
        f.write("SELF-PRUNING NEURAL NETWORK - EXPERIMENTAL RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write(f"  Lambda values: {lambda_values}\n")
        f.write(f"  Number of epochs: {num_epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n\n")
        
        f.write("RESULTS TABLE:\n")
        f.write(f"{'Lambda':<15} {'Accuracy (%)':<20} {'Sparsity (%)':<20}\n")
        f.write("-" * 70 + "\n")
        for result in results:
            f.write(f"{result['lambda']:<15.1e} {result['accuracy']:<20.2f} {result['sparsity']:<20.2f}\n")
        f.write("="*70 + "\n\n")
        
        f.write("OBSERVATIONS:\n")
        f.write("1. Higher λ forces more pruning (higher sparsity)\n")
        f.write("2. More aggressive pruning reduces accuracy\n")
        f.write("3. Optimal λ depends on application requirements\n")
        f.write("4. This trade-off is fundamental to neural network pruning\n")
    
    print(" Results saved to 'results.txt'")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. results.txt - Experimental results")
    print("  2. gate_distributions.png - Histogram of gate values")
    print("  3. training_metrics.png - Training curves")
    print("\n")
