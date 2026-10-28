"""Model and Trainer classes for ALA."""
from collections import defaultdict, Counter
from dataclasses import dataclass, replace
from itertools import product
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


class FF(nn.Module):
    """Network model with functions for forward-pass and predictions."""
    def __init__(self, database_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(database_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward-pass with two hidden layers."""
        # First hidden
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)

        # Second hidden
        out = self.fc2(out)
        out = self.gelu(out)
        out = self.dropout(out)

        out = self.fc_out(out)

        return out

    def predict(self, vector, language, idx2fam, device):
        """Predicts based on new data."""
        vector = torch.Tensor(np.array([vector[language][2]]))
        vector = vector.to(device)

        outs = self(vector)
        _, prediction = torch.max(outs.data, 1)
        prediction = idx2fam[prediction.item()]

        return prediction


@dataclass
class TrainConfig:
    """
    Stores the hyperparameters for the model.
    Default values correspond to the best results for Lexibank.
    """
    epochs: int = 5000
    batch: int = 8
    hidden: int = 3
    lr: float = 1e-3
    patience: int = 50
    dropout: float = 0.3


@dataclass
class TrainingContext:
    """Stores optimizer/scheduler/criterion for the model."""
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler


@dataclass
class ExperimentData:
    """Everything prepared once by prepare_data(), shared across all runs."""
    tensor_ds: TensorDataset
    idx2fam: dict
    class_weights: torch.Tensor


@dataclass
class DataLoaders:
    """Storing the train/test/dev split."""
    train: DataLoader
    val: DataLoader
    test: DataLoader


@dataclass
class RunSettings:
    """Everything about how/where to run training, independent of the data itself."""
    config: TrainConfig
    device: str
    checkpoint_path: str = 'results/best-mpar.pt'


def prepare_data(data, test_langs, device):
    """One-time data preparation: labels, weights, tensors. Runs once, not per-run."""
    idx2fam = dict(enumerate(set(data[lang][0] for lang in data)))
    fam2idx = {fam: idx for idx, fam in idx2fam.items()}

    tests = {lang: data[lang] for lang in data if lang in test_langs}
    features = [data[lang][2] for lang in data if lang not in test_langs]
    labels = [fam2idx[data[lang][0]] for lang in data if lang not in test_langs]

    summary_stats = {
        'Number of families': len(fam2idx),
        'Number of languages': len(data),
        'Size of vector': len(features[0]),
    }
    print(summary_stats)

    fam2w = Counter(data[lang][0] for lang in data)

    class_weights = torch.FloatTensor(
        [fam2w[max(fam2w, key=fam2w.get)] / fam2w[fam] for fam in fam2w]
        ).to(device)

    features = torch.Tensor(np.array(features))
    all_labels = torch.LongTensor(np.array(labels))
    tensor_ds = TensorDataset(features, all_labels)

    return tensor_ds, idx2fam, fam2w, class_weights, tests


def evaluate(model, test_loader, idx2fam):
    """
    Evaluate model, returning per-family accuracy and balanced accuracy.
    This deactivates the drop-out of neurons from the training.
    """
    model.eval()
    family_results = defaultdict(list)

    with torch.no_grad():
        for test_data, labels in test_loader:
            outputs = model(test_data)
            _, preds = torch.max(outputs.data, 1)
            for label, pred in zip(labels, preds):
                family_results[int(label)].append(int(pred))

    fam_avg = {}
    for fam, preds in family_results.items():
        corr = sum(1 for pred in preds if fam == pred)
        total = len(preds)
        fam_avg[idx2fam[fam]] = [100 * corr / total, total]

    bal_acc = sum(v[0] for v in fam_avg.values()) / len(fam_avg)
    model.train()

    return fam_avg, bal_acc


class ModelTrainer:
    """The whole context for training the model."""
    def __init__(self, exp_data: ExperimentData, settings: RunSettings):
        self.exp_data = exp_data
        self.settings = settings

        self.loaders = self._split_data()
        self.bundle = self._build_model()

    def _split_data(self) -> DataLoaders:
        tensor_ds = self.exp_data.tensor_ds
        device = self.settings.device
        batch = self.settings.config.batch

        train_ds, temp = train_test_split(tensor_ds, test_size=0.4, stratify=tensor_ds.tensors[1])
        val_ds, test_ds = train_test_split(temp, test_size=0.5, stratify=[l for _, l in temp])

        train_ds = [(x.to(device), y.to(device)) for x, y in train_ds]
        val_ds = [(x.to(device), y.to(device)) for x, y in val_ds]
        test_ds = [(x.to(device), y.to(device)) for x, y in test_ds]

        return DataLoaders(
            train=DataLoader(train_ds, batch_size=batch, shuffle=True),
            val=DataLoader(val_ds, batch_size=batch),
            test=DataLoader(test_ds, batch_size=batch)
        )

    def _build_model(self) -> TrainingContext:
        config = self.settings.config
        device = self.settings.device
        input_dim = self.exp_data.tensor_ds.tensors[0].size(1)
        output_dim = len(self.exp_data.idx2fam)

        model = FF(
            input_dim, config.hidden * output_dim, output_dim, dropout=config.dropout,
            ).to(device)

        criterion = nn.CrossEntropyLoss(weight=self.exp_data.class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=config.patience, factor=0.2, mode='max')

        return TrainingContext(model, criterion, optimizer, scheduler)

    def train(self):
        """Train the model."""
        no_improve = 0
        best_macro = 0
        idx2fam = self.exp_data.idx2fam

        for epoch in range(self.settings.config.epochs):
            if no_improve >= 50:
                break

            for train_data, labels in self.loaders.train:
                self.bundle.optimizer.zero_grad()
                outputs = self.bundle.model(train_data)
                loss = self.bundle.criterion(outputs, labels)
                loss.backward()
                self.bundle.optimizer.step()

            _, bal_acc = evaluate(self.bundle.model, self.loaders.val, idx2fam)
            print(f'Epoch: {epoch}. Loss: {loss.item():.5f}. Bal. acc.: {round(bal_acc, 5)}.')

            if bal_acc > best_macro:
                best_macro = bal_acc
                no_improve = 0
                torch.save(self.bundle.model.state_dict(), self.settings.checkpoint_path)
            else:
                no_improve += 1

            self.bundle.scheduler.step(bal_acc)


    def evaluate_final(self, tests):
        """Evaluate the final model with test-set and all neurons."""
        idx2fam = self.exp_data.idx2fam
        device = self.settings.device

        self.bundle.model.load_state_dict(
            torch.load(self.settings.checkpoint_path, weights_only=True)
        )
        f_fam_avg, f_total = evaluate(self.bundle.model, self.loaders.test, idx2fam)

        predictions = {
            (lang, tests[lang][0]): self.bundle.model.predict(tests, lang, idx2fam, device)
            for lang in tests
        }

        return f_fam_avg, f_total, predictions


def run_model(data, runs, config: TrainConfig, test_langs={}):
    """Runs the whole workflow, based on the ModelTrainer class."""
    device = 'mps' if torch.backends.mps.is_available() \
    else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Current device:', device)

    res_per_fam = defaultdict(list)
    results_experiment = defaultdict(list)

    tensor_ds, idx2fam, fam2w, class_weights, tests = prepare_data(data, test_langs, device)
    exp_data = ExperimentData(tensor_ds, idx2fam, class_weights)
    settings = RunSettings(config, device)

    for run in range(runs):
        print(f'--- New Run: {run + 1}/{runs} ---')

        trainer = ModelTrainer(exp_data, settings)
        trainer.train()
        f_fam_avg, f_total, predictions = trainer.evaluate_final(tests)

        for key, pred in predictions.items():
            results_experiment[key].append(pred)

        for fam, res in f_fam_avg.items():
            res_per_fam[fam].append([run + 1, fam, fam2w[fam], res[1], round(res[0], 3)])

        res_per_fam['TOTAL'].append([run + 1, 'TOTAL', len(data), len(data)/5, round(f_total, 3)])

    return res_per_fam, results_experiment


def sweep_grid(data, base_config, param_grid, test_langs={}, runs=10):
    """
    Run train() across all combinations of hyperparameters in param_grid.
    param_grid: {'lr': [], 'batch': [], 'hidden': []}
    """
    param_names = list(param_grid.keys())
    value_combinations = list(product(*param_grid.values()))

    sweep_results = {}

    for combo in value_combinations:
        overrides = dict(zip(param_names, combo))
        config = replace(base_config, **overrides)

        label = ', '.join(f'{k}={v}' for k, v in overrides.items())
        print(f'=== Testing {label} ===')

        result_per_fam, _ = run_model(data, runs, config, test_langs=test_langs)

        total_rows = result_per_fam['TOTAL']
        accs = [row[4] for row in total_rows]
        avg_acc = sum(accs) / len(accs)

        sweep_results[combo] = {
            'params': overrides,
            'avg_acc': avg_acc,
            'per_run_acc': accs,
        }
        print(f'{label}: {avg_acc:.2f} avg bal_acc (runs: {[round(a, 2) for a in accs]})')

    return sweep_results, param_names
