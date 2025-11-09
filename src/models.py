import torchimport torch"""Models and optimizer for the deepfake detection pipeline.

import torch.nn as nn

import torch.nn.functional as Fimport torch.nn as nn



class DeepfakeDetector(nn.Module):import torch.nn.functional as FProvides a baseline RandomForest classifier and a simple Genetic Algorithm

    def __init__(self):

        super().__init__()to optimize hyperparameters (bio-inspired optimizer).

        

        # Convolutional layersclass DeepfakeDetector(nn.Module):"""

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    def __init__(self):from typing import Tuple, Dict, Any

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)        super().__init__()import numpy as np

        self.dropout = nn.Dropout(0.25)

                

        # Fully connected layers

        self.fc1 = nn.Linear(128 * 16 * 4, 512)        # Convolutional layersfrom sklearn.ensemble import RandomForestClassifier

        self.fc2 = nn.Linear(512, 64) 

        self.fc3 = nn.Linear(64, 2)  # 2 classes: real/fake        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)from sklearn.model_selection import cross_val_score

        

    def forward(self, x):        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)from sklearn.metrics import accuracy_score

        # Input shape: (batch, channels, height, width)

        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pool        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        x = self.dropout(x)

                self.pool = nn.MaxPool2d(2, 2)

        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> Pool

        x = self.dropout(x)        self.dropout = nn.Dropout(0.25)class BaselineDetector:

        

        x = self.pool(F.relu(self.conv3(x)))  # Conv -> ReLU -> Pool            def __init__(self, params: Dict[str, Any] = None):

        x = self.dropout(x)

                # Fully connected layers        if params is None:

        # Flatten

        x = x.view(-1, 128 * 16 * 4)        self.fc1 = nn.Linear(128 * 16 * 4, 512)            params = {"n_estimators": 100, "max_depth": None, "random_state": 42}

        

        # Fully connected layers        self.fc2 = nn.Linear(512, 64)         self.params = params

        x = F.relu(self.fc1(x))

        x = self.dropout(x)        self.fc3 = nn.Linear(64, 2)  # 2 classes: real/fake        self.clf = RandomForestClassifier(**self.params)

        

        x = F.relu(self.fc2(x))        

        x = self.dropout(x)

            def forward(self, x):    def train(self, X: np.ndarray, y: np.ndarray):

        x = self.fc3(x)

                # Input shape: (batch, channels, height, width)        self.clf = RandomForestClassifier(**self.params)

        return F.log_softmax(x, dim=1)
        x = self.pool(F.relu(self.conv1(x)))  # Conv -> ReLU -> Pool        self.clf.fit(X, y)

        x = self.dropout(x)        return self

        

        x = self.pool(F.relu(self.conv2(x)))  # Conv -> ReLU -> Pool    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        x = self.dropout(x)        return self.clf.predict_proba(X)

        

        x = self.pool(F.relu(self.conv3(x)))  # Conv -> ReLU -> Pool    def predict(self, X: np.ndarray) -> np.ndarray:

        x = self.dropout(x)        return self.clf.predict(X)

        

        # Flatten

        x = x.view(-1, 128 * 16 * 4)class GeneticOptimizer:

            """A small Genetic Algorithm to optimize RandomForest hyperparameters.

        # Fully connected layers

        x = F.relu(self.fc1(x))    Chromosome encoding: [n_estimators, max_depth_code]

        x = self.dropout(x)    - n_estimators in [10, 300]

            - max_depth_code maps to {None, 5, 10, 20, 40}

        x = F.relu(self.fc2(x))    """

        x = self.dropout(x)

            MAX_DEPTH_CHOICES = [None, 5, 10, 20, 40]

        x = self.fc3(x)

            def __init__(self, pop_size: int = 12, generations: int = 10, mutation_prob: float = 0.2):

        return F.log_softmax(x, dim=1)        self.pop_size = pop_size
        self.generations = generations
        self.mutation_prob = mutation_prob

    def _random_chrom(self):
        n_estimators = np.random.randint(10, 301)
        max_depth_code = np.random.randint(0, len(self.MAX_DEPTH_CHOICES))
        return (n_estimators, max_depth_code)

    def _decode(self, chrom):
        n_estimators, max_depth_code = chrom
        return {"n_estimators": int(n_estimators), "max_depth": self.MAX_DEPTH_CHOICES[int(max_depth_code)], "random_state": 42}

    def _fitness(self, chrom, X, y):
        params = self._decode(chrom)
        clf = RandomForestClassifier(**params)
        # use 3-fold CV to estimate accuracy
        try:
            scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray):
        # initialize population
        pop = [self._random_chrom() for _ in range(self.pop_size)]
        pop_fitness = [self._fitness(c, X, y) for c in pop]

        for gen in range(self.generations):
            # selection: tournament
            new_pop = []
            for _ in range(self.pop_size // 2):
                # pick parents
                i1, i2 = np.random.randint(0, self.pop_size, size=2)
                p1 = pop[i1] if pop_fitness[i1] > pop_fitness[i2] else pop[i2]
                i3, i4 = np.random.randint(0, self.pop_size, size=2)
                p2 = pop[i3] if pop_fitness[i3] > pop_fitness[i4] else pop[i4]

                # crossover (single point simple swap)
                child1 = (p1[0], p2[1])
                child2 = (p2[0], p1[1])

                # mutation
                if np.random.rand() < self.mutation_prob:
                    child1 = (np.clip(child1[0] + np.random.randint(-20, 21), 10, 300), np.random.randint(0, len(self.MAX_DEPTH_CHOICES)))
                if np.random.rand() < self.mutation_prob:
                    child2 = (np.clip(child2[0] + np.random.randint(-20, 21), 10, 300), np.random.randint(0, len(self.MAX_DEPTH_CHOICES)))

                new_pop.extend([child1, child2])

            # evaluate new population
            new_fitness = [self._fitness(c, X, y) for c in new_pop]

            # elitism: keep the best individuals from combined
            combined = pop + new_pop
            combined_f = pop_fitness + new_fitness
            idx = np.argsort(combined_f)[::-1][:self.pop_size]
            pop = [combined[i] for i in idx]
            pop_fitness = [combined_f[i] for i in idx]

        # return best decoded params and its fitness
        best_idx = int(np.argmax(pop_fitness))
        best_chrom = pop[best_idx]
        best_params = self._decode(best_chrom)
        return best_params, pop_fitness[best_idx]


if __name__ == "__main__":
    print("models.py quick smoke test")
    # tiny synthetic test
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    opt = GeneticOptimizer(pop_size=6, generations=4)
    best_params, best_score = opt.optimize(X, y)
    print("best:", best_params, best_score)
