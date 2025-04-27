import random
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from numba import njit
from pathlib import Path

import json, os


def load_pgm(pgm_path: str) -> np.ndarray:
    """
    Lê PGM no formato P5 (binário) e retorna array uint8 ou uint16.
    """
    with open(pgm_path, 'rb') as f:
        # cabeçalho
        magic = f.readline().strip()
        if magic != b'P5':
            raise ValueError("Só PGM P5 suportado")
        # pula comentários
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        # resolução
        width, height = map(int, line.split())
        maxval = int(f.readline())
        # lê pixels
        dtype = np.uint8 if maxval < 256 else np.uint16
        img = np.fromfile(f, dtype=dtype, count=width*height)
    return img.reshape((height, width))


def load_summary(run_dir: str) -> Dict[str, float]:
    """Lê *.json com as métricas finais."""
    path = os.path.join(run_dir, "files", "wandb-summary.json")
    return json.load(open(path)) if os.path.isfile(path) else {}

def sample_history(run_dir: str, metric: str, k: int = 10) -> List[Tuple[int, float]]:
    """
    Devolve k amostras uniformes (step, valor) da métrica pedida.
    Não usa pandas.
    """
    hist = os.path.join(run_dir, "files", "wandb-history.jsonl")
    steps, vals = [], []
    with open(hist) as f:
        for line in f:
            d = json.loads(line)
            if metric in d:
                steps.append(d["_step"])
                vals.append(d[metric])

    if not steps:
        return []

    idx = np.linspace(0, len(steps)-1, k, dtype=int)
    return [(steps[i], vals[i]) for i in idx]


def _parse_simple_yaml(path: str) -> dict:
    root, stack, indents = {}, [None], [-1]
    current = root

    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        key, _, value = line.lstrip().partition(":")
        key = key.strip()
        value = value.strip()

        while indent <= indents[-1]:
            stack.pop()
            indents.pop()
        if stack[-1] is None:
            current = root
        else:
            current = stack[-1]

        if not value:
            current[key] = {}
            stack.append(current[key])
            indents.append(indent)
        else:
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                value = [float(x) if x.replace(".", "", 1).isdigit() else x for x in value]
            elif value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value) if value.isdigit() else float(value)
                except ValueError:
                    pass
            current[key] = value

    return root

@njit
def normalize_module(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value


@njit
def distance_to_goal(
    x: float, y: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    dist = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist >= max_value:
        return max_value
    else:
        return dist


# @njit # !!!!!!
def angle_to_goal(
    x: float, y: float, theta: float, goal_x: float, goal_y: float, max_value: float
) -> float:
    o_t = np.array([np.cos(theta), np.sin(theta)])
    g_t = np.array([goal_x - x, goal_y - y])

    cross_product = np.cross(o_t, g_t)
    dot_product = np.dot(o_t, g_t)

    alpha = np.abs(np.arctan2(np.linalg.norm(cross_product), dot_product))

    if alpha >= max_value:
        return max_value

    else:
        return alpha


@njit
def min_laser(measurement: np.ndarray, threshold: float):
    laser = np.min(measurement)
    if laser <= threshold:
        return True, laser
    else:
        return False, laser


@njit
def uniform_random(min_val, max_val):
    return np.random.uniform(min_val, max_val)


@njit
def uniform_random_int(min_val, max_val):
    return np.random.randint(min_val, max_val + 1)


def safe_stats(data):
    clean_data = [v for v in data if v is not None]
    if not clean_data:
        return 0.0, 0.0, 0.0
    return np.mean(clean_data), np.min(clean_data), np.max(clean_data)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_info(info: dict) -> dict:
    """Mantém apenas as chaves desejadas e converte arrays em listas."""
    valid_keys = {
        "obstacle_score",
        "orientation_score",
        "progress_score",
        "time_score",
        "action",
        "dist",
        "alpha",
        "min_lidar",
        "max_lidar",
    }

    cleaned = {}
    for key in valid_keys:
        if key in info:
            val = info[key]
            # Se tiver algum valor que é array, converta para lista
            if isinstance(val, np.ndarray):
                val = val.tolist()
            cleaned[key] = val

    return cleaned


class CustomMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale_ = None
        self.min_ = None

    def fit(self, X):
        X = np.array(X)
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        data_range = self.data_max - self.data_min
        data_range[data_range == 0] = 1
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min * self.scale_
        return self

    def transform(self, X):
        X = np.array(X)
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Property:
    def __init__(self):
        self.dimension = 2


class Index:
    def __init__(self, properties: Optional[Property] = None):
        self.properties = properties if properties is not None else Property()
        self.items = []  # cada item é uma tupla: (id, bbox, obj opcional)

    def insert(self, id: int, bbox: Tuple[float, float, float, float], obj=None):
        """
        Insere um item no índice.
        bbox: (xmin, ymin, xmax, ymax)
        """
        self.items.append((id, bbox, obj))

    def intersection(
        self, bbox: Tuple[float, float, float, float], objects: bool = False
    ) -> List:
        """
        Retorna os ids (ou objetos, se objects=True) dos itens que intersectam com o bbox.
        """
        results = []
        for id, ibox, obj in self.items:
            if self._intersect(ibox, bbox):
                results.append(obj if objects else id)
        return results

    def _intersect(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> bool:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
        if xmax1 < xmin2 or xmax2 < xmin1:
            return False
        if ymax1 < ymin2 or ymax2 < ymin1:
            return False
        return True

    def __iter__(self):
        for id, bbox, obj in self.items:
            yield id


def print_config_table(config_dict):
    table_width = 45
    horizontal_line = "-" * table_width
    print(horizontal_line)

    for config_name, config_values in config_dict.items():
        print(f"| {config_name + '/':<41} |")
        print(horizontal_line)
        if isinstance(config_values, dict):
            for key, value in config_values.items():
                print(f"|    {key.ljust(20)} | {str(value).ljust(15)} |")
        else:
            print(f"|    {str(config_values).ljust(20)} | {' '.ljust(15)}|")
        print(horizontal_line)
