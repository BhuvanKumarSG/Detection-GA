import os
import joblib
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import random


class BaselineModel:
    def __init__(self, model_path: str = 'models/baseline.ckpt', random_state: int = 42):
        self.model_path = model_path
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)

    def train(self, X, y):
        self.model.fit(X, y)
        self.save_model()

    def predict_proba(self, X):
        # return probability for positive class
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False


class GAOptimizedModel(BaselineModel):
    def __init__(self, model_path: str = 'models/ga_optimized.ckpt', random_state: int = 42):
        super().__init__(model_path=model_path, random_state=random_state)

    def _random_params(self):
        # wider, more expressive search space
        return {
            'n_estimators': random.randint(10, 500),
            'max_depth': random.choice([None] + list(range(5, 101, 5))),
            'max_features': random.choice(['sqrt', 'log2', None]),
            'min_samples_split': random.randint(2, 20),
            'min_samples_leaf': random.randint(1, 20),
            'bootstrap': random.choice([True, False]),
            'criterion': random.choice(['gini', 'entropy'])
        }

    def _mutate(self, params):
        p = params.copy()
        key = random.choice(list(p.keys()))
        if key == 'n_estimators':
            p[key] = max(10, min(500, p[key] + random.randint(-50, 50)))
        elif key == 'max_depth':
            choices = [None] + list(range(5, 101, 5))
            p[key] = random.choice(choices)
        elif key == 'max_features':
            p[key] = random.choice(['sqrt', 'log2', None])
        elif key == 'min_samples_split':
            p[key] = max(2, min(50, p[key] + random.randint(-5, 5)))
        elif key == 'min_samples_leaf':
            p[key] = max(1, min(50, p[key] + random.randint(-3, 3)))
        elif key == 'bootstrap':
            p[key] = not p[key]
        elif key == 'criterion':
            p[key] = 'entropy' if p[key] == 'gini' else 'gini'
        return p

    def _crossover(self, a, b):
        child = {}
        for k in a.keys():
            child[k] = a[k] if random.random() < 0.5 else b[k]
        return child

    def _evaluate(self, params, X, y, cv=3, scoring='roc_auc'):
        # use stratified splits for small datasets
        clf = RandomForestClassifier(random_state=self.random_state, **params)
        try:
            skf = StratifiedKFold(n_splits=max(2, cv), shuffle=True, random_state=self.random_state)
            scores = cross_val_score(clf, X, y, cv=skf, scoring=scoring, n_jobs=1)
            return float(scores.mean())
        except Exception:
            return 0.0

    def optimize(self, X, y, generations: int = 12, population_size: int = 20,
                 cv: int = 3, scoring: str = 'roc_auc', elitism: int = 2,
                 mutation_rate: float = 0.35, tournament_k: int = 3):
        # enhanced GA with elitism and tournament selection
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        # initialize population
        population = [self._random_params() for _ in range(population_size)]
        scores = [self._evaluate(p, X, y, cv=cv, scoring=scoring) for p in population]

        for gen in range(generations):
            # keep elites
            ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            elites = [p for p, s in ranked[:elitism]]

            # produce next generation
            new_pop = elites.copy()

            # tournament selection helper
            def tournament_select(pop, sc, k=tournament_k):
                ids = random.sample(range(len(pop)), min(k, len(pop)))
                best = max(ids, key=lambda i: sc[i])
                return pop[best]

            while len(new_pop) < population_size:
                parent_a = tournament_select(population, scores)
                parent_b = tournament_select(population, scores)
                child = self._crossover(parent_a, parent_b)
                if random.random() < mutation_rate:
                    child = self._mutate(child)
                new_pop.append(child)

            population = new_pop
            scores = [self._evaluate(p, X, y, cv=cv, scoring=scoring) for p in population]

        # best params
        best_idx = int(np.argmax(scores))
        best_params = population[best_idx]

        # set model with best params and train on full data
        self.model = RandomForestClassifier(random_state=self.random_state, **best_params)
        self.model.fit(X, y)
        self.save_model()
        return best_params