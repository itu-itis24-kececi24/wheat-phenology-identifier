import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import unicodedata

import itertools
import random
import os
import json
import pickle
import pandas as pd
from typing import List, Tuple


# Make cv2 optional for environments without OpenCV
try:
    import cv2
except Exception:
    cv2 = None

class WheatFramework:
    def __init__(self, excel_path, root_dir):
        self.excel_path = os.path.abspath(excel_path)
        self.root_dir = os.path.abspath(root_dir)
        self.le = LabelEncoder()
        self.df = None
        
        # We define the windows based on your Excel columns (expected names)
        self.expected_stage_cols = ['1-Ekim', '2 - Çıkış', '3 - Çimlenme', '4 - Kardeşlenme', 
                    '5 - Sapa Kalkma', '6 - Başaklanma', '7 - Çiçeklenme', 
                    '8 - Olgunlaşma', '9 - Hasat']
        self.stages = [
            ('1-Ekim', '2 - Çıkış', 'PS0'),
            ('2 - Çıkış', '3 - Çimlenme', 'PS1'),
            ('3 - Çimlenme', '4 - Kardeşlenme', 'PS2'),
            ('4 - Kardeşlenme', '5 - Sapa Kalkma', 'PS3'),
            ('5 - Sapa Kalkma', '6 - Başaklanma', 'PS4'),
            ('6 - Başaklanma', '7 - Çiçeklenme', 'PS5'),
            ('7 - Çiçeklenme', '8 - Olgunlaşma', 'PS6'),
            ('8 - Olgunlaşma', '9 - Hasat', 'PS7')
        ]

    def _normalize(self, s):
        if s is None:
            return ''
        s = str(s).lower()
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        # keep only alnum
        s = ''.join(ch for ch in s if ch.isalnum())
        return s

    def _map_columns(self, df):
        # Map expected column names to actual columns in df using normalized matching
        col_map = {}
        norm_cols = {self._normalize(c): c for c in df.columns}
        for exp in self.expected_stage_cols:
            n = self._normalize(exp)
            if n in norm_cols:
                col_map[exp] = norm_cols[n]
            else:
                # try contains match
                found = None
                for nc, orig in norm_cols.items():
                    if n in nc or nc in n:
                        found = orig
                        break
                if found:
                    col_map[exp] = found
                else:
                    # give up and keep exp as-is (will coerce to NaT later)
                    col_map[exp] = exp
        # Apply renaming where needed
        rename_dict = {v: k for k, v in col_map.items() if v in df.columns and v != k}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        return df, col_map
    
    def _get_dates_from_excel(self):
        # Load data
        df = pd.read_excel(self.excel_path)

        # Map/normalize column names so accented characters don't break matching
        df, col_map = self._map_columns(df)
        
        # Convert Excel date columns (DD.MM.YYYY) to Python Datetime
        all_cols = self.expected_stage_cols
        for col in all_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            else:
                df[col] = pd.NaT
        
        return df

    def find_station_path(self, station_raw):
        """Return (station_path, station_folder) for a Station Code value."""
        s_folder = None
        try:
            s_numeric = f"{float(station_raw):05.2f}"
        except Exception:
            s_numeric = str(station_raw).strip()

        variants = [
            s_numeric,
            str(s_numeric).replace('.', '_'),
            str(s_numeric).replace('.', ','),
            str(s_numeric).lstrip('0'),
        ]

        for v in variants:
            p = os.path.join(self.root_dir, v)
            if os.path.exists(p):
                return p, v

        if os.path.isdir(self.root_dir):
            for folder in os.listdir(self.root_dir):
                if str(folder).replace('.', '').replace('_', '') == str(s_numeric).replace('.', ''):
                    return os.path.join(self.root_dir, folder), folder

        return None, s_folder

    def get_stage_boundaries(self, row):
        """Return phenology boundary dates in the expected column order."""
        return [row.get(col) for col in self.expected_stage_cols]

    def label_for_date(self, row, img_date):
        """Assign the original hard PS label for an image date using framework stages."""
        for start_col, end_col, stage_label in self.stages:
            s_date = row.get(start_col)
            e_date = row.get(end_col)
            if pd.notna(s_date) and pd.notna(e_date) and s_date <= img_date < e_date:
                return stage_label

        harvest = row.get('9 - Hasat')
        if pd.notna(harvest) and img_date == harvest:
            return 'PS7'

        return None

    def analyze_stagewise_day_diffs(self):
        df = self._get_dates_from_excel()
        stationwise_counts = {stage_label: {} for _, _, stage_label in self.stages}
        for _, row in df.iterrows():
            station_raw = row.get('Station Code')
            # Build a set of station-folder variants to try
            s_folder = None
            try:
                s_numeric = f"{float(station_raw):05.2f}"
            except Exception:
                s_numeric = str(station_raw).strip()

            variants = [s_numeric, str(s_numeric).replace('.', '_'), str(s_numeric).replace('.', ','), str(s_numeric).lstrip('0')]

            for v in variants:
                p = os.path.join(self.root_dir, v)
                if os.path.exists(p):
                    s_folder = v
                    break

            for start_col, end_col, stage_label in self.stages:
                s_date = row.get(start_col)
                e_date = row.get(end_col)
                stationwise_counts[stage_label][f"{s_folder}_{row.get('Year')}"] = {"start": s_date, "end": e_date, "days": (e_date - s_date).days if pd.notna(s_date) and pd.notna(e_date) else None}

        return pd.DataFrame(stationwise_counts)
            
    def get_dataframe(self):
        if self.df is None:
            self.df = self._load_and_preprocess()
        return self.df
    
    def get_stagewise_missing_data(self):
        df_data = self.get_dataframe()
        df_days = self.analyze_stagewise_day_diffs()
        stations = df_data['station_year'].unique()

        missing_data_per_station = {}
        missing_dates_for_stations = {}
        for stage in df_days.columns:
            print(f"\nStage: {stage}")
            for station in stations:
                if station not in missing_data_per_station.keys():
                    missing_data_per_station[station] = {}
                count = df_data[df_data['station_year'] == station][df_data['label'] == stage].shape[0]
                days = df_days.loc[df_days.index == station, stage].values[0]
                start_date = df_days.loc[df_days.index == station, stage].values[0]['start']
                end_date = df_days.loc[df_days.index == station, stage].values[0]['end']
                days_count = days["days"]

                missing_data_per_station[station][stage] = {"missing_count": days_count - count if days_count is not None else None, "total_days": days_count, "data_count": count, "days": days}

                if days_count is not None and count < days_count:
                    total_dates = set(pd.date_range(start_date, end_date))
                    existing_dates = set(pd.to_datetime(df_data[(df_data['station_year'] == station) & (df_data['label'] == stage)]['path'].apply(lambda x: re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})', x).groups() if re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})', x) else None).dropna().apply(lambda x: pd.to_datetime(f"{x[0]}-{x[1]}-{x[2]}") if x[0] else pd.to_datetime(f"{x[3]}-{x[4]}-{x[5]}"))))
                    missing_dates = total_dates - existing_dates
                    missing_dates_for_stations[station] = missing_dates
        return missing_data_per_station, missing_dates_for_stations



    def _load_and_preprocess(self):
        # Load data
        df = self._get_dates_from_excel()

        data_list = []
        # Accept multiple date separators/formats: YYYY_MM_DD, YYYY-MM-DD, or YYYYMMDD
        date_pattern = re.compile(r'(\d{4})[_-](\d{2})[_-](\d{2})|(\d{4})(\d{2})(\d{2})')

        print(f"Searching for images in: {self.root_dir}")

        for _, row in df.iterrows():
            station_raw = row.get('Station Code')
            # Build a set of station-folder variants to try
            s_folder = None
            try:
                s_numeric = f"{float(station_raw):05.2f}"
            except Exception:
                s_numeric = str(station_raw).strip()

            variants = [s_numeric, str(s_numeric).replace('.', '_'), str(s_numeric).replace('.', ','), str(s_numeric).lstrip('0')]

            station_path = None
            for v in variants:
                p = os.path.join(self.root_dir, v)
                if os.path.exists(p):
                    station_path = p
                    s_folder = v
                    break

            if station_path is None:
                # Try fuzzy match (ignore separators)
                for folder in os.listdir(self.root_dir):
                    if str(folder).replace('.', '').replace('_', '') == str(s_numeric).replace('.', ''):
                        station_path = os.path.join(self.root_dir, folder)
                        s_folder = folder
                        break

            if station_path is None:
                print(f"Station folder not found for Excel value '{station_raw}' (tried variants).")
                continue

            planting = row.get('1-Ekim')
            harvest = row.get('9 - Hasat')

            # Walk year/K/magnification folders to be robust to layout
            for year_dir in os.listdir(station_path):
                year_path = os.path.join(station_path, year_dir)
                if not os.path.isdir(year_path):
                    continue

                for kfold in os.listdir(year_path):
                    k_path = os.path.join(year_path, kfold)

                    if not os.path.isdir(k_path):
                        continue
                    if 'k1' not in k_path.lower():
                        continue  # skip unless k1 (camera 1). If you want to include k2, remove this check.

                    for mag in os.listdir(k_path):
                        img_dir = os.path.join(k_path, mag)
                        if not os.path.isdir(img_dir):
                            continue
                        if '10x' in img_dir.lower():
                            continue  # skip x10 folders if present, focus on x1

                        for img_name in os.listdir(img_dir):
                            full_path = os.path.join(img_dir, img_name)
                            match = date_pattern.search(img_name)
                            if not match:
                                continue
                            
                            print(f"Found image: {full_path} (matched date in filename) - ", end='')
                            if match.group(1):
                                print(f"  Parsed date 1: {match.group(1)}-{match.group(2)}-{match.group(3)}")
                                y, m, d = match.group(1), match.group(2), match.group(3)
                            else:
                                print(f"  Parsed date 2: {match.group(4)}-{match.group(5)}-{match.group(6)}")
                                y, m, d = match.group(4), match.group(5), match.group(6)

                            try:
                                img_date = pd.to_datetime(f"{y}-{m}-{d}")
                            except Exception:
                                continue

                            if pd.notna(planting) and pd.notna(harvest):
                                if planting <= img_date <= harvest:
                                    assigned_label = None
                                    for start_col, end_col, stage_label in self.stages:
                                        s_date = row.get(start_col)
                                        e_date = row.get(end_col)
                                        if pd.notna(s_date) and pd.notna(e_date) and s_date <= img_date < e_date:
                                            assigned_label = stage_label
                                            break

                                    if not assigned_label and pd.notna(harvest) and img_date == harvest:
                                        assigned_label = 'PS7'

                                    if assigned_label:
                                        data_list.append({
                                            'path': full_path,
                                            'label': assigned_label,
                                            'group_id': row.get('ID'),
                                            'station_year': f"{s_folder}_{row.get('Year', year_dir)}"
                                        })

        final_df = pd.DataFrame(data_list)
        if final_df.empty:
            print("Handshake failed. Check Station Code formatting, Excel date columns and image filename dates.")
        else:
            print(f"Success! Matched {len(final_df)} images.")
        return final_df

    def get_splits(self, meta_df, n_splits=5):
        X = meta_df['path'].values
        y = self.le.fit_transform(meta_df['label'])
        groups = meta_df['group_id'].values
        gkf = GroupKFold(n_splits=n_splits)
        return X, y, list(gkf.split(X, y, groups=groups))

    def get_sample(self, X, y):
        idx = np.random.randint(0, len(X))
        if cv2 is not None:
            img = cv2.imread(X[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            import matplotlib.image as mpimg
            img = mpimg.imread(X[idx])
            # mpimg may return float image in [0,1], convert if necessary
            if img.dtype != np.uint8 and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        plt.figure(figsize=(10,5))
        plt.imshow(img)
        plt.title(f"X: {os.path.basename(X[idx])}\nY: {self.le.inverse_transform([y[idx]])[0]}")
        plt.axis('off')
        plt.show()


# Utilities: generate group-folds (8 train / 2 test), save models, save results, and runner template
# The user provides a `trainer_fn(train_df, test_df, fold_id)` that trains a model,
# returns a dictionary with keys: 'model' (optional), 'predictions' (DataFrame or list), 'metrics' (dict)

def generate_group_folds(meta_df: pd.DataFrame,
                         group_col: str = 'group_id',
                         n_train: int = 8,
                         n_test: int = 2,
                         num_folds: int = None,
                         random_state: int = None) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index pairs by selecting n_test groups as test and the rest as train.

    Returns a list of (train_idx_array, test_idx_array) where indices are integer positions
    into meta_df.

    If the number of all possible combinations C(G, n_test) is larger than `num_folds`,
    a random subset of combinations is returned (seeded by `random_state`).

    Example usage:
      folds = generate_group_folds(meta_df, n_train=8, n_test=2, num_folds=10, random_state=42)
      for i, (train_idx, test_idx) in enumerate(folds, 1):
          train_df = meta_df.iloc[train_idx]
          test_df  = meta_df.iloc[test_idx]

    """
    groups = list(pd.unique(meta_df[group_col]))
    groups = sorted(groups)
    G = len(groups)
    if n_train + n_test > G:
        raise ValueError(f"Not enough groups ({G}) for n_train={n_train} + n_test={n_test}")

    all_test_combs = list(itertools.combinations(groups, n_test))
    total_combs = len(all_test_combs)
    if num_folds is None or num_folds >= total_combs:
        chosen = all_test_combs
    else:
        rng = random.Random(random_state)
        chosen = rng.sample(all_test_combs, num_folds)

    folds = []
    for test_groups in chosen:
        test_set = set(test_groups)
        train_groups = [g for g in groups if g not in test_set]
        train_idx = meta_df.index[meta_df[group_col].isin(train_groups)].to_numpy()
        test_idx = meta_df.index[meta_df[group_col].isin(test_groups)].to_numpy()
        folds.append((train_idx, test_idx))

    return folds


def save_model(obj, path: str):
    """
    Save a model object to `path`. Handles common frameworks:
      - PyTorch (state_dict or whole model if given and torch available)
      - Keras/TensorFlow (model.save)
      - otherwise: pickle dump

    The function will try framework-specific save functions where possible.
    """
    # Try PyTorch
    try:
        import torch
        if hasattr(obj, 'state_dict'):
            torch.save(obj.state_dict(), path)
            return
        elif hasattr(torch, 'save') and hasattr(obj, '__class__'):
            # fallback: try saving whole object (may not be ideal)
            torch.save(obj, path)
            return
    except Exception:
        pass

    # Try Keras / TensorFlow
    try:
        # Keras models have `save` method
        if hasattr(obj, 'save'):
            obj.save(path)
            return
    except Exception:
        pass

    # Default: pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_results(predictions, metrics: dict, out_dir: str, fold_id: int):
    """
    Save predictions and metrics for a fold.
    - `predictions` can be a Pandas DataFrame (preferred) or a list/dict.
    - `metrics` is a serializable dict of scalar values.

    Files written to `out_dir/fold_{fold_id}/`:
      - predictions.csv (if DataFrame) or predictions.json
      - metrics.json
    """
    os.makedirs(out_dir, exist_ok=True)
    fold_dir = os.path.join(out_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    # Save predictions
    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(os.path.join(fold_dir, 'predictions.csv'), index=False)
    else:
        with open(os.path.join(fold_dir, 'predictions.json'), 'w', encoding='utf8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Save metrics
    with open(os.path.join(fold_dir, 'metrics.json'), 'w', encoding='utf8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def run_fold_with_trainer(meta_df: pd.DataFrame,
                          train_idx,
                          test_idx,
                          trainer_fn,
                          out_dir: str = 'results',
                          fold_id: int = 1,
                          save_model_flag: bool = True):
    """
    Run a single fold using a user-provided `trainer_fn`.

    `trainer_fn(train_df, test_df, fold_id)` must:
      - train the user's model on `train_df` (a DataFrame with at minimum the `path` and `label` columns),
      - evaluate/predict on `test_df`,
      - return a dict with keys:
          - 'model' (optional): trained model object
          - 'predictions': DataFrame or list/dict with predictions
          - 'metrics': dict of evaluation metrics (scalars)

    The function handles saving model + results to `out_dir/fold_{fold_id}`.
    """
    train_df = meta_df.iloc[train_idx].reset_index(drop=True)
    test_df = meta_df.iloc[test_idx].reset_index(drop=True)

    result = trainer_fn(train_df, test_df, fold_id)
    if result is None:
        raise RuntimeError('trainer_fn must return a dict with keys: predictions and metrics (model optional)')

    model = result.get('model')
    predictions = result.get('predictions')
    metrics = result.get('metrics', {})

    fold_out = os.path.join(out_dir, f'fold_{fold_id}')
    os.makedirs(fold_out, exist_ok=True)

    if save_model_flag and model is not None:
        model_path = os.path.join(fold_out, 'model.pth')
        save_model(model, model_path)

    save_results(predictions, metrics, out_dir, fold_id)

    return result



    
