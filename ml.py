import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


# ==================== Utility Functions ====================

def cyclic_shift(img, shift_h, shift_w):
    """
    Cyclic shift: shift right by shift_w (wrap around to left), shift down by shift_h (wrap around to top)
    """
    return torch.roll(img, shifts=(int(shift_h), int(shift_w)), dims=(-2, -1))


def circular_permute_features(features_list, shift_amount):
    """
    Perform circular permutation on 16 features

    shift_amount=0: [f0, f1, f2, ..., f15]
    shift_amount=1: [f1, f2, f3, ..., f15, f0]
    ...
    """
    n = len(features_list)
    shift_amount = shift_amount % n
    return features_list[shift_amount:] + features_list[:shift_amount]


# ==================== Data Loading ====================

def load_mnist_data():
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_db_full = torchvision.datasets.MNIST('mnist_data', train=True,
                                               download=True, transform=transform)
    train_db, val_db = torch.utils.data.random_split(train_db_full, [50000, 10000])
    test_db = torchvision.datasets.MNIST('mnist_data/', train=False,
                                         download=True, transform=transform)

    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ==================== Model Definition ====================

class StandardCNN(nn.Module):
    """Standard CNN baseline"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(7*7*32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 7*7*32)
        return self.fc(x)


class EquivariantCNN(nn.Module):
    """
    Group Equivariant CNN

    Each layer performs equivariant operations on the 16 group elements.
    """
    def __init__(self, num_classes=10, step_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.step_size = step_size

        # Quotient group structure
        self.grid_size = 28 // step_size  # 4
        self.n_group = self.grid_size ** 2  # 16

        # Representatives
        self.representatives = []
        for h in range(self.grid_size):
            for w in range(self.grid_size):
                self.representatives.append((h * step_size, w * step_size))

        # Kernel offsets (7×7)
        self.kernel_offsets = []
        for h in range(step_size):
            for w in range(step_size):
                self.kernel_offsets.append((h, w))

        self.kernel_size = len(self.kernel_offsets)  # 49

        print(f"  Number of representatives: {self.n_group}")
        print(f"  Kernel size: {self.kernel_size}")
        print(f"  Layer 0 computation: {self.n_group} × {self.kernel_size} = {self.n_group * self.kernel_size}")

        # Network layers (no BN, to preserve equivariance)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16 * self.n_group, 32, 3, padding=1)  # Input is concatenation of 16 features
        self.conv3 = nn.Conv2d(32 * self.n_group, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(7*7*32, num_classes)
        self.relu = nn.ReLU()

    def layer0_project(self, x):
        """
        Layer 0: Project onto reduced space

        For each representative:
        1. Perform 49 shifts
        2. Pass each shifted version through CNN
        3. Average the 49 results

        Input: [B, 1, 28, 28]
        Output: list of 16 feature maps, each [B, 16, 28, 28]
        """
        outputs = []

        # First convolution (optimization: leverage translation equivariance)
        feat_base = self.relu(self.conv1(x))  # [B, 16, 28, 28]

        for (rep_h, rep_w) in self.representatives:
            # For this representative, collect 49 shifted versions
            shifted_feats = []

            for (off_h, off_w) in self.kernel_offsets:
                # Total shift = representative + offset
                total_h = rep_h + off_h
                total_w = rep_w + off_w

                # Shift on feature map (leverage equivariance)
                feat_shifted = cyclic_shift(feat_base, total_h, total_w)
                shifted_feats.append(feat_shifted)

            # Average over 49
            shifted_feats = torch.stack(shifted_feats, dim=0)  # [49, B, 16, 28, 28]
            avg_feat = shifted_feats.mean(dim=0)  # [B, 16, 28, 28]

            outputs.append(avg_feat)

        return outputs  # list of 16, each [B, 16, 28, 28]

    def equivariant_layer(self, features_list, conv_layer):
        """
        Equivariant layer operation

        For each of the 16 circular permutations of features, pass through convolution.

        Input: list of 16 features
        Output: list of 16 new features
        """
        outputs = []

        # For 16 circular permutations
        for perm_idx in range(self.n_group):
            # Circularly permute features
            permuted = circular_permute_features(features_list, perm_idx)

            # Concatenate all features
            x_cat = torch.cat(permuted, dim=1)  # [B, 16×C, H, W]

            # Pass through convolution
            out = self.relu(conv_layer(x_cat))

            outputs.append(out)

        return outputs  # list of 16

    def forward(self, x):
        """
        Complete forward pass

        Layer 0: 784 → 16 features
        Layer 1: 16 permutations → 16 features
        Layer 2: 16 permutations → 16 features
        Layer 3: classification
        """
        # Layer 0: Projection
        features = self.layer0_project(x)  # list of 16 [B, 16, H, W]

        # Layer 1: Equivariant
        features = self.equivariant_layer(features, self.conv2)  # list of 16 [B, 32, H, W]

        # Layer 2: Equivariant
        features = self.equivariant_layer(features, self.conv3)  # list of 16 [B, 32, H, W]

        # Pooling and FC
        outputs = []
        for feat in features:
            x_pooled = self.pool(feat)  # [B, 32, 7, 7]
            x_flat = x_pooled.view(-1, 7*7*32)
            out = self.fc(x_flat)  # [B, num_classes]
            outputs.append(out)

        return outputs  # list of 16 [B, num_classes]


# ==================== Training Functions ====================

def train_standard_cnn(model, train_loader, val_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_state = None

    print("Training standard CNN...")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        epoch_time = time.time() - start_time

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Time={epoch_time:.1f}s")

    model.load_state_dict(best_state)
    print(f"Training completed! Best Val Acc: {best_val_acc:.4f}\n")
    return model


def train_equivariant_cnn(model, train_loader, val_loader, epochs=5, lr=1e-3):
    """
    Train Equivariant CNN

    Key points:
    - To ensure all 16 positions learn digit classification capability, compute digit cross-entropy for all 16 positions for each sample and average (equivalent to flattening (B,16,num_classes) to (B*16, num_classes) for cross entropy, with labels repeated 16 times)
    - Keep random translation as data augmentation for robustness
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_state = None

    print("Training Equivariant CNN ")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        start_time = time.time()

        for batch_idx, (x_orig, labels) in enumerate(train_loader):
            x_orig, labels = x_orig.to(device), labels.to(device)

            # Randomly translate images (data augmentation)
            batch_size = x_orig.size(0)
            shifts_h = torch.randint(0, 28, (batch_size,))
            shifts_w = torch.randint(0, 28, (batch_size,))

            x = torch.stack([
                cyclic_shift(x_orig[i:i+1], shifts_h[i].item(), shifts_w[i].item())
                for i in range(batch_size)
            ]).squeeze(1)

            optimizer.zero_grad()

            # Forward
            logits_list = model(x)  # list of 16 [B, num_classes]
            logits = torch.cat(logits_list, dim=1)  # [B, 16*num_classes]

            # Reshape to [B, 16, num_classes]
            logits_reshaped = logits.view(-1, model.n_group, model.num_classes)

            # Digit loss: compute cross-entropy for all 16 positions (ensure each position is supervised)
            # Method: flatten (B,16,num_classes) to (B*16, num_classes), repeat labels 16 times
            B = logits_reshaped.size(0)
            digit_loss = F.cross_entropy(logits_reshaped.view(-1, model.num_classes),
                                         labels.repeat_interleave(model.n_group))

            loss = digit_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, DigitLoss: {digit_loss.item():.4f}")

        avg_loss = epoch_loss / batch_count
        epoch_time = time.time() - start_time

        # Validation (no random translation)
        model.eval()
        val_digit_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)

                logits_list = model(x)
                logits = torch.cat(logits_list, dim=1)
                # Digit prediction: take max over all 16 positions
                digit_pred = torch.argmax(logits, dim=1) % model.num_classes
                val_digit_correct += (digit_pred == labels).sum().item()
                val_total += len(labels)

        val_digit_acc = val_digit_correct / val_total

        if val_digit_acc > best_val_acc:
            best_val_acc = val_digit_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch}: AvgLoss={avg_loss:.4f}, ValDigitAcc={val_digit_acc:.4f}, Time={epoch_time/60:.1f}min")
        print("-" * 80)

    model.load_state_dict(best_state)
    print(f"Training completed! Best digit accuracy: {best_val_acc:.4f}\n")
    return model


# ==================== Test / Evaluation Functions ====================

def test_model(model, test_loader, is_equivariant=False, transform_fn=None):
    """
    Test model

    Returns:
    - digit_acc: digit recognition accuracy
    """
    model.eval()
    digit_correct = 0
    total = 0

    with torch.no_grad():
        for x, labels in test_loader:
            if transform_fn:
                x = transform_fn(x)
            x, labels = x.to(device), labels.to(device)

            if is_equivariant:
                logits_list = model(x)
                logits = torch.cat(logits_list, dim=1)  # [B, 16*10]

                # Digit prediction: take max over all 16 positions
                digit_pred = torch.argmax(logits, dim=1) % model.num_classes
            else:
                logits = model(x)
                digit_pred = torch.argmax(logits, dim=1)

            digit_correct += (digit_pred == labels).sum().item()
            total += len(labels)

    digit_acc = digit_correct / total
    return digit_acc, None


def test_equivariance_property(model, test_loader, base_shift=(3, 5)):
    """
    Verify equivariance property

    Test: For translation (a,b), after translation (7k, 7m):
    - Digit recognition accuracy should be the same (or similar)
    (removed position prediction / position accuracy output)
    """
    model.eval()
    print(f"\nVerifying Equivariance Property (base translation={base_shift}):")
    print("-" * 80)

    # Test different translations by multiples of 7
    grid_shifts = [(0, 7), (14, 0), (7, 7), (21, 7), (28, 0), (28, 14)]

    results = []

    for (grid_h, grid_w) in grid_shifts:
        total_shift_h = base_shift[0] + grid_h
        total_shift_w = base_shift[1] + grid_w

        transform_fn = lambda x: cyclic_shift(x.to(device), total_shift_h, total_shift_w)

        digit_correct = 0
        total = 0

        with torch.no_grad():
            for x, labels in test_loader:
                x = transform_fn(x)
                x, labels = x.to(device), labels.to(device)

                logits_list = model(x)
                logits = torch.cat(logits_list, dim=1)

                # Digit prediction
                digit_pred = torch.argmax(logits, dim=1) % model.num_classes
                digit_correct += (digit_pred == labels).sum().item()

                total += len(labels)

        digit_acc = digit_correct / total

        results.append({
            'shift': (total_shift_h, total_shift_w),
            'grid_shift': (grid_h, grid_w),
            'digit_acc': digit_acc,
        })

        print(f"  Shift({total_shift_h:2d},{total_shift_w:2d}) [+grid({grid_h},{grid_w})]: "
              f"Digit accuracy={digit_acc:.4f}")

    # Analysis
    digit_accs = [r['digit_acc'] for r in results]
    avg_acc = np.mean(digit_accs)
    std_acc = np.std(digit_accs)

    print(f"\n  Digit recognition accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  Conclusion: {'✓ Equivariance maintained well' if std_acc < 0.01 else '⚠ Some deviation present'}")

    return results


# ==================== Main Program ====================

def main():
    print("="*80)
    print("True Group Equivariant CNN Experiment")
    print("="*80)
    print()

    # Load data
    train_loader, val_loader, test_loader = load_mnist_data()

    # Train standard CNN
    print("\n1. Standard CNN")
    print("="*80)
    standard_cnn = StandardCNN().to(device)
    print(f"Number of parameters: {sum(p.numel() for p in standard_cnn.parameters()):,}\n")
    standard_cnn = train_standard_cnn(standard_cnn, train_loader, val_loader, epochs=2)

    # Train Equivariant CNN (only digit supervision, all positions supervised)
    print("\n2. Equivariant CNN")
    print("="*80)
    equi_cnn = EquivariantCNN().to(device)
    print(f"Number of parameters: {sum(p.numel() for p in equi_cnn.parameters()):,}\n")
    equi_cnn = train_equivariant_cnn(equi_cnn, train_loader, val_loader, epochs=2)

    # ==================== Test A: Robustness to different ranges of translation ====================
    print("\n3. Test A: Robustness to different ranges of translation")
    print("="*80)

    test_groups = {
        "Small shifts (0-5 pixels)": [(1,1), (2,1), (3,3), (4,2), (5,3)],
        "Medium shifts (7-12 pixels)": [(8,9), (11,5), (9,7), (12,3), (10,8)],
        "Large shifts (18-24 pixels)": [(21,12), (15,5), (18,18), (24,7)],
        "Random translations": "random"
    }

    print(f"\n{'Test Category':<25s} {'Standard CNN':<15s} {'Equivariant CNN':<15s} {'Improvement':<10s}")
    print("-"*75)

    for group_name, shifts in test_groups.items():
        if shifts == "random":
            std_accs = []
            equi_accs = []

            for _ in range(5):
                sh = np.random.randint(0, 28)
                sw = np.random.randint(0, 28)
                transform = lambda x: cyclic_shift(x.to(device), sh, sw)

                acc_std, _ = test_model(standard_cnn, test_loader, False, transform)
                acc_equi, _ = test_model(equi_cnn, test_loader, True, transform)

                std_accs.append(acc_std)
                equi_accs.append(acc_equi)

            avg_std = np.mean(std_accs)
            avg_equi = np.mean(equi_accs)
            diff = avg_equi - avg_std

            print(f"{group_name:<25s} {avg_std:.4f}         {avg_equi:.4f}           {diff:+.4f}")
        else:
            std_accs = []
            equi_accs = []

            for (sh, sw) in shifts:
                transform = lambda x, sh=sh, sw=sw: cyclic_shift(x.to(device), sh, sw)

                acc_std, _ = test_model(standard_cnn, test_loader, False, transform)
                acc_equi, _ = test_model(equi_cnn, test_loader, True, transform)

                std_accs.append(acc_std)
                equi_accs.append(acc_equi)

            avg_std = np.mean(std_accs)
            avg_equi = np.mean(equi_accs)
            diff = avg_equi - avg_std

            print(f"{group_name:<25s} {avg_std:.4f}         {avg_equi:.4f}           {diff:+.4f}")

    # ==================== Test B: Equivariance property verification====================
    print("\n" + "="*80)
    print("4. Test B: Equivariance property verification")
    print("="*80)

    # Test a few different base translations
    for base_shift in [(3, 5), (2, 4), (5, 1)]:
        test_equivariance_property(equi_cnn, test_loader, base_shift)

    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)

    print("""
Key conclusions:
1. Explicit position prediction is now removed; only digit recognition remains.
2. To ensure the model does not only learn one or two positions, we supervise all 16 positions simultaneously for digits during training (compute cross-entropy for each position).
3. Random translation (data augmentation) is still retained for robustness.
    """)


if __name__ == "__main__":
    main()