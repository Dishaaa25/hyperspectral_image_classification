import matplotlib
matplotlib.use('Agg')  # <<< FIX: non-interactive backend, no plt.show() hang

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, roc_curve, auc,
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, time

# ── Constants ───────────────────────────────────────────────────────────────
PATCH_SIZE = 15
PCA_COMPONENTS = 32
NUM_TRANSFORMER_LAYERS = 3
NUM_HEADS = 4
TOKEN_DIM = 64
DROPOUT_RATE = 0.3
RANDOM_SEEDS = [42, 123, 456]
TRAINING_RATIOS = [0.05, 0.10, 0.15]

RESULTS_DIR = "reviewer_results"
MODELS_DIR = "saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Data Loading ────────────────────────────────────────────────────────────
def load_dataset(name="indian_pines", data_path=None, gt_path=None):
    from scipy.io import loadmat
    if name == "indian_pines":
        data = loadmat("Indian_pines_corrected.mat")["indian_pines_corrected"]
        labels = loadmat("Indian_pines_gt.mat")["indian_pines_gt"]
    elif name == "pavia_university":
        data = loadmat("pavia/PaviaU.mat")["paviaU"]
        labels = loadmat("pavia/PaviaU_gt.mat")["paviaU_gt"]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    print(f"[{name}] Data: {data.shape}, Labels: {labels.shape}")
    return data, labels


# ── Preprocessing ───────────────────────────────────────────────────────────
def preprocess_data(data):
    h, w, b = data.shape
    reshaped = data.reshape(-1, b).astype(np.float64)  # <<< FIX: float64
    scaler = StandardScaler()
    normalized = scaler.fit_transform(reshaped)
    return normalized.reshape(h, w, b)


def apply_pca(data, n_components=PCA_COMPONENTS):
    h, w, b = data.shape
    reshaped = data.reshape(-1, b).astype(np.float64)  # <<< FIX: float64
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped)
    print(f"  PCA explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    return reduced.reshape(h, w, n_components).astype(np.float32)  # back to 32 for TF


# ── Patch Extraction ────────────────────────────────────────────────────────
def create_patches(data, labels, patch_size=PATCH_SIZE):
    h, w, b = data.shape
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    patches, plabels = [], []
    for i in range(h):
        for j in range(w):
            if labels[i, j] == 0:
                continue
            p = padded[i:i+patch_size, j:j+patch_size, :]
            if p.shape[0] == patch_size and p.shape[1] == patch_size:
                patches.append(p)
                plabels.append(labels[i, j])
    return np.array(patches, dtype=np.float32), np.array(plabels)


# ── Train/Test Split ────────────────────────────────────────────────────────
def prepare_data(data, labels, patch_size=PATCH_SIZE,
                 train_ratio=None, samples_per_class=None, seed=42):
    np.random.seed(seed)
    patches, plabels = create_patches(data, labels, patch_size)
    unique = np.unique(plabels)
    unique = unique[unique != 0]
    num_classes = len(unique)
    label_map = {c: i for i, c in enumerate(unique)}
    mapped = np.array([label_map[l] for l in plabels])

    Xtr, ytr, Xte, yte = [], [], [], []
    for ci in range(num_classes):
        idx = np.where(mapped == ci)[0]
        np.random.shuffle(idx)
        if samples_per_class is not None:
            n = min(samples_per_class, len(idx) - 1)
        else:
            n = max(1, int(len(idx) * train_ratio))
        Xtr.extend(patches[idx[:n]]); ytr.extend(mapped[idx[:n]])
        Xte.extend(patches[idx[n:]]); yte.extend(mapped[idx[n:]])

    Xtr, ytr = np.array(Xtr), np.array(ytr)
    Xte, yte = np.array(Xte), np.array(yte)
    print(f"  Train: {len(Xtr)}, Test: {len(Xte)}, Classes: {num_classes}")
    return Xtr, to_categorical(ytr, num_classes), Xte, to_categorical(yte, num_classes), label_map


class DynamicResizeLayer(layers.Layer):
    """Resize first tensor to match spatial dims of second tensor."""
    def call(self, inputs):
        tensor, ref = inputs
        return tf.image.resize(tensor, [tf.shape(ref)[1], tf.shape(ref)[2]])

    def get_config(self):
        return super().get_config()


class PositionalEncodingBroadcast(layers.Layer):
    def __init__(self, height, width, dim, **kwargs):
        super().__init__(**kwargs)
        self.height, self.width, self.dim = height, width, dim

    def build(self, input_shape):
        pos = self._sinusoidal(self.height * self.width, self.dim)
        self.pos = tf.Variable(initial_value=pos, trainable=True)

    def call(self, x):
        return x + self.pos

    def _sinusoidal(self, positions, d):
        pos = tf.range(positions, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d, dtype=tf.float32)[tf.newaxis, :]
        angles = pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d, tf.float32))
        sines = tf.math.sin(angles[:, 0::2])
        cosines = tf.math.cos(angles[:, 1::2])
        return tf.cast(tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...], tf.float32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"height": self.height, "width": self.width, "dim": self.dim})
        return cfg


def build_spectral_reduction(inputs, channels=32):
    shape = inputs.shape
    x = layers.Reshape((shape[1]*shape[2], shape[3]))(inputs)
    c1 = layers.Conv1D(channels//2, 3, padding='same', activation='relu')(x)
    c3 = layers.Conv1D(channels//2, 7, padding='same', activation='relu')(x)
    c5 = layers.Conv1D(channels//2, 11, padding='same', activation='relu')(x)
    multi = layers.Concatenate()([c1, c3, c5])
    multi = layers.Conv1D(channels, 1, padding='same')(multi)
    skip = layers.Conv1D(channels, 1, padding='same')(x)
    out = layers.Add()([multi, skip])
    out = layers.Activation('relu')(out)
    return layers.Reshape((shape[1], shape[2], channels))(out)


def build_spectral_reduction_single(inputs, channels=32):
    """Single-scale fallback for ablation."""
    shape = inputs.shape
    x = layers.Reshape((shape[1]*shape[2], shape[3]))(inputs)
    x = layers.Conv1D(channels, 7, padding='same', activation='relu')(x)
    return layers.Reshape((shape[1], shape[2], channels))(x)


def build_transformer_encoder(inputs, heads=NUM_HEADS, dim=TOKEN_DIM,
                              num_layers=NUM_TRANSFORMER_LAYERS):
    shape = inputs.shape
    h, w = shape[1], shape[2]
    x = layers.Reshape((h * w, shape[3]))(inputs)
    x = layers.Dense(dim)(x)
    initial = x
    x = PositionalEncodingBroadcast(h, w, dim)(x)
    skips = []
    for i in range(num_layers):
        skips.append(x)
        att = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)(x, x, x)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(dim * 4, activation="gelu")(x)
        ffn = layers.Dense(dim)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        if i > 0:
            x = layers.Add()([x, skips[i - 1]])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Add()([x, initial])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return layers.Reshape((h, w, dim))(x)


def _safe_concat(upsampled, skip_connection):
    """
    <<< FIX: Always resize upsampled to match skip_connection before concat.
    Handles ALL odd-dimension mismatches (15->7->3->6≠7, etc.)
    """
    up = DynamicResizeLayer()([upsampled, skip_connection])
    return layers.Concatenate(axis=3)([skip_connection, up])


def build_unet(inputs):
    """U-Net with safe concat at every decoder stage."""
    # ── Encoder ──
    c1 = layers.Conv2D(128, 3, padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1); c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(128, 3, padding='same')(c1)
    c1 = layers.BatchNormalization()(c1); c1 = layers.Activation('relu')(c1)
    s0 = layers.Conv2D(128, 1, padding='same')(inputs)
    c1 = layers.Add()([c1, s0])
    p1 = layers.MaxPooling2D(2)(c1)       # 15->7

    c2 = layers.Conv2D(256, 3, padding='same')(p1)
    c2 = layers.BatchNormalization()(c2); c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(256, 3, padding='same')(c2)
    c2 = layers.BatchNormalization()(c2); c2 = layers.Activation('relu')(c2)
    s1 = layers.Conv2D(256, 1, padding='same')(p1)
    c2 = layers.Add()([c2, s1])
    p2 = layers.MaxPooling2D(2)(c2)       # 7->3

    # ── Bottleneck ──
    c3 = layers.Conv2D(512, 3, padding='same')(p2)
    c3 = layers.BatchNormalization()(c3); c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(512, 3, padding='same')(c3)
    c3 = layers.BatchNormalization()(c3); c3 = layers.Activation('relu')(c3)
    s2 = layers.Conv2D(512, 1, padding='same')(p2)
    c3 = layers.Add()([c3, s2])

    # ── Decoder 1: upsample bottleneck -> concat with c2 ──
    u1 = layers.UpSampling2D(2)(c3)      # 3->6
    u1 = layers.Conv2D(256, 2, activation='relu', padding='same')(u1)
    m1 = _safe_concat(u1, c2)            # <<< FIX: resize 6->7 then concat

    c4 = layers.Conv2D(256, 3, padding='same')(m1)
    c4 = layers.BatchNormalization()(c4); c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(256, 3, padding='same')(c4)
    c4 = layers.BatchNormalization()(c4); c4 = layers.Activation('relu')(c4)
    s3 = layers.Conv2D(256, 1, padding='same')(m1)
    c4 = layers.Add()([c4, s3])

    # ── Decoder 2: upsample c4 -> concat with c1 ──
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c4)  # 7->14
    u2 = layers.BatchNormalization()(u2); u2 = layers.Activation('relu')(u2)
    m2 = _safe_concat(u2, c1)            # <<< FIX: resize 14->15 then concat

    c5 = layers.Conv2D(128, 3, padding='same')(m2)
    c5 = layers.BatchNormalization()(c5); c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(128, 3, padding='same')(c5)
    c5 = layers.BatchNormalization()(c5); c5 = layers.Activation('relu')(c5)
    s4 = layers.Conv2D(128, 1, padding='same')(m2)
    c5 = layers.Add()([c5, s4])

    # Deep skip from input
    deep = layers.Conv2D(128, 1, padding='same')(inputs)
    out = layers.Add()([c5, deep])
    out = layers.BatchNormalization()(out)
    return layers.Activation('relu')(out)


def build_classification_head(inputs, num_classes, dropout=DROPOUT_RATE):
    x = layers.Conv2D(64, 1)(inputs)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64, 1)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    pooled = layers.Concatenate()([gap, gmp])
    pooled = layers.Dropout(dropout)(pooled)
    return layers.Activation('softmax')(layers.Dense(num_classes)(pooled))


def build_full_model(input_shape, num_classes,
                     heads=NUM_HEADS, dim=TOKEN_DIM,
                     num_trans_layers=NUM_TRANSFORMER_LAYERS,
                     spectral_channels=32, dropout=DROPOUT_RATE,
                     use_transformer=True, use_unet=True,
                     use_ensemble=True, use_multiscale=True):
    inp = layers.Input(shape=input_shape)

    # Spectral reduction
    if use_multiscale:
        sr = build_spectral_reduction(inp, channels=spectral_channels)
    else:
        sr = build_spectral_reduction_single(inp, channels=spectral_channels)

    skip_inp = layers.Conv2D(spectral_channels, 1, padding='same')(inp)
    feat = layers.Add()([sr, skip_inp])

    # Transformer
    if use_transformer:
        t_out = build_transformer_encoder(feat, heads, dim, num_trans_layers)
        t_skip = layers.Conv2D(dim, 1, padding='same')(feat)
        feat = layers.Add()([t_out, t_skip])
    else:
        shape = feat.shape
        x = layers.Reshape((shape[1]*shape[2], shape[3]))(feat)
        x = layers.Dense(dim, activation='relu')(x)
        feat = layers.Reshape((shape[1], shape[2], dim))(x)

    # U-Net
    if use_unet:
        u_out = build_unet(feat)
        u_skip = layers.Conv2D(128, 1, padding='same')(feat)
        feat = layers.Add()([u_out, u_skip])
    else:
        feat = layers.Conv2D(128, 3, padding='same', activation='relu')(feat)
        feat = layers.Conv2D(128, 3, padding='same', activation='relu')(feat)

    # Classification
    if use_ensemble:
        out = build_classification_head(feat, num_classes, dropout)
    else:
        x = layers.Conv2D(64, 1, activation='relu')(feat)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Activation('softmax')(layers.Dense(num_classes)(x))

    return models.Model(inputs=inp, outputs=out)


def train_and_evaluate(model, X_tr, y_tr, X_te, y_te,
                       epochs=10, batch_size=16, lr=5e-4):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    t0 = time.time()
    hist = model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs,
                     validation_data=(X_te, y_te), callbacks=[es], verbose=1)
    train_time = time.time() - t0

    t0 = time.time()
    y_prob = model.predict(X_te, verbose=0)
    inf_time = time.time() - t0

    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_te, axis=1)

    m = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'train_time_s': train_time,
        'inference_time_s': inf_time,
        'y_prob': y_prob, 'y_true': y_true, 'y_pred': y_pred,
        'history': hist.history,
    }
    print(f"  OA={m['accuracy']:.4f}  F1={m['f1']:.4f}  κ={m['kappa']:.4f}  "
          f"Train={train_time:.1f}s  Inf={inf_time:.1f}s")
    return m


def save_model(model, name, metrics=None):
    """Save model weights + optional metrics JSON."""
    path = os.path.join(MODELS_DIR, name)
    model.save(path + ".keras")
    print(f"  Model saved: {path}.keras")
    if metrics:
        # Strip non-serializable fields
        saveable = {k: v for k, v in metrics.items()
                    if k not in ('y_prob', 'y_true', 'y_pred', 'confusion_matrix', 'history')}
        saveable['history_keys'] = list(metrics.get('history', {}).keys())
        with open(path + "_metrics.json", 'w') as f:
            import json; json.dump(saveable, f, indent=2, default=str)
    # Save history separately as numpy
    if metrics and 'history' in metrics:
        np.savez(path + "_history.npz", **{k: np.array(v) for k, v in metrics['history'].items()})
    # Save predictions
    if metrics and 'y_prob' in metrics:
        np.savez(path + "_predictions.npz",
                 y_prob=metrics['y_prob'],
                 y_true=metrics['y_true'],
                 y_pred=metrics['y_pred'],
                 confusion_matrix=metrics['confusion_matrix'])
    print(f"  Artifacts saved: {path}_metrics.json, _history.npz, _predictions.npz")


def load_saved_model(name):
    """Load a previously saved model."""
    path = os.path.join(MODELS_DIR, name + ".keras")
    custom_objects = {
        'DynamicResizeLayer': DynamicResizeLayer,
        'PositionalEncodingBroadcast': PositionalEncodingBroadcast,
    }
    model = models.load_model(path, custom_objects=custom_objects)
    print(f"  Model loaded: {path}")

    # Load metrics if available
    metrics = {}
    metrics_path = os.path.join(MODELS_DIR, name + "_metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
    pred_path = os.path.join(MODELS_DIR, name + "_predictions.npz")
    if os.path.exists(pred_path):
        preds = np.load(pred_path)
        metrics['y_prob'] = preds['y_prob']
        metrics['y_true'] = preds['y_true']
        metrics['y_pred'] = preds['y_pred']
        metrics['confusion_matrix'] = preds['confusion_matrix']
    hist_path = os.path.join(MODELS_DIR, name + "_history.npz")
    if os.path.exists(hist_path):
        h = np.load(hist_path)
        metrics['history'] = {k: h[k].tolist() for k in h.files}

    return model, metrics


def complexity_report(model, input_shape):
    params = model.count_params()
    flops = None
    try:
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        inp = tf.TensorSpec(shape=(1, *input_shape), dtype=tf.float32)
        concrete = tf.function(model).get_concrete_function(inp)
        opts = ProfileOptionBuilder.float_operation()
        flops = profile(concrete.graph, options=opts).total_float_ops
    except Exception:
        pass
    r = {'params': params, 'params_M': round(params/1e6, 2),
         'FLOPs': flops, 'FLOPs_G': round(flops/1e9, 2) if flops else "N/A"}
    print(f"  Params: {r['params_M']}M | FLOPs: {r['FLOPs_G']}G")
    return r


def noise_robustness_test(model, X_te, y_te, snr_levels=[40, 30, 20, 10]):
    results = []
    y_true = np.argmax(y_te, axis=1)
    for snr in snr_levels:
        sig_power = np.mean(X_te ** 2)
        noise_power = sig_power / (10 ** (snr / 10))
        X_noisy = X_te + np.random.normal(0, np.sqrt(noise_power), X_te.shape).astype(np.float32)
        y_pred = np.argmax(model.predict(X_noisy, verbose=0), axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        kap = cohen_kappa_score(y_true, y_pred)
        results.append({'SNR_dB': snr, 'OA': acc, 'F1': f1, 'Kappa': kap})
        print(f"  SNR={snr}dB: OA={acc:.4f} F1={f1:.4f} κ={kap:.4f}")
    return pd.DataFrame(results)


def plot_roc_curves(y_true, y_prob, num_classes, label_map, save_path=None):
    y_true_bin = to_categorical(y_true, num_classes)
    inv_map = {v: k for k, v in label_map.items()}
    plt.figure(figsize=(12, 10))
    auc_scores = {}
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[inv_map.get(i, i)] = roc_auc
        plt.plot(fpr, tpr, label=f"Class {inv_map.get(i,i)} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Per-Class ROC Curves'); plt.legend(fontsize=7, loc='lower right')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()  # <<< FIX: close instead of show
    return auc_scores


def plot_training_curves(history, title="", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
    ax1.set_title(f'{title} Loss'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax2.plot(history['accuracy'], label='Train'); ax2.plot(history['val_accuracy'], label='Val')
    ax2.set_title(f'{title} Accuracy'); ax2.legend(); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()  # <<< FIX


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.close()  # <<< FIX


def run_hyperparameter_table():
    hp = pd.DataFrame([
        {"Hyperparameter": "Patch Size",            "Search Range": "{11, 13, 15}",         "Selected": PATCH_SIZE},
        {"Hyperparameter": "PCA Components",         "Search Range": "{16, 32, 64}",         "Selected": PCA_COMPONENTS},
        {"Hyperparameter": "Spectral Conv Channels",  "Search Range": "{16, 32, 64}",         "Selected": 32},
        {"Hyperparameter": "Conv Kernel Sizes",       "Search Range": "{3}, {3,7}, {3,7,11}", "Selected": "{3, 7, 11}"},
        {"Hyperparameter": "Transformer Layers",      "Search Range": "{1, 2, 3, 4}",         "Selected": NUM_TRANSFORMER_LAYERS},
        {"Hyperparameter": "Attention Heads",          "Search Range": "{2, 4, 8}",            "Selected": NUM_HEADS},
        {"Hyperparameter": "Token Dimension",          "Search Range": "{32, 64, 128}",        "Selected": TOKEN_DIM},
        {"Hyperparameter": "Learning Rate",            "Search Range": "{1e-3, 5e-4, 1e-4}",   "Selected": "5e-4"},
        {"Hyperparameter": "Batch Size",               "Search Range": "{16, 32, 64, 128}",    "Selected": 16},
        {"Hyperparameter": "Dropout Rate",             "Search Range": "{0.1, 0.2, 0.3, 0.5}", "Selected": DROPOUT_RATE},
        {"Hyperparameter": "Optimizer",                "Search Range": "{Adam, AdamW, SGD}",    "Selected": "Adam"},
        {"Hyperparameter": "Epochs (max)",             "Search Range": "30",                    "Selected": 30},
        {"Hyperparameter": "Early Stopping Patience",  "Search Range": "{5, 10, 15}",           "Selected": 10},
    ])
    hp.to_csv(f"{RESULTS_DIR}/hyperparameter_table.csv", index=False)
    print(hp.to_string(index=False))
    return hp


def run_main_experiment(data_path, gt_path, dataset_name="indian_pines"):
    print(f"\n{'='*60}\n  MAIN EXPERIMENT: {dataset_name}\n{'='*60}")

    data, labels = load_dataset(dataset_name, data_path, gt_path)
    preprocessed = preprocess_data(data)
    reduced = apply_pca(preprocessed, PCA_COMPONENTS)

    X_tr, y_tr, X_te, y_te, lmap = prepare_data(
        reduced, labels, PATCH_SIZE, train_ratio=0.80, seed=42)
    input_shape = X_tr.shape[1:]
    num_classes = y_tr.shape[1]

    model = build_full_model(input_shape, num_classes)
    model.summary()

    metrics = train_and_evaluate(model, X_tr, y_tr, X_te, y_te)

    # Save model + all artifacts
    save_model(model, f"{dataset_name}_main", metrics)

    # Plots (all saved to disk, no blocking)
    plot_training_curves(metrics['history'], title=dataset_name,
                         save_path=f"{RESULTS_DIR}/{dataset_name}_curves.png")
    plot_confusion_matrix(metrics['confusion_matrix'],
                          save_path=f"{RESULTS_DIR}/{dataset_name}_cm.png")
    auc_scores = plot_roc_curves(
        metrics['y_true'], metrics['y_prob'], num_classes, lmap,
        save_path=f"{RESULTS_DIR}/{dataset_name}_roc.png")

    comp = complexity_report(model, input_shape)

    print("\n── Noise Robustness ──")
    noise_df = noise_robustness_test(model, X_te, y_te)
    noise_df.to_csv(f"{RESULTS_DIR}/{dataset_name}_noise.csv", index=False)

    return model, metrics, comp, noise_df, auc_scores


def run_low_label_experiments(data_path, gt_path, dataset_name="indian_pines"):
    print(f"\n{'='*60}\n  LOW-LABEL EXPERIMENTS: {dataset_name}\n{'='*60}")

    data, labels = load_dataset(dataset_name, data_path, gt_path)
    preprocessed = preprocess_data(data)
    reduced = apply_pca(preprocessed, PCA_COMPONENTS)

    all_results = []
    for ratio in TRAINING_RATIOS:
        print(f"\n── Ratio: {ratio*100:.0f}% ──")
        seed_results = []
        for seed in RANDOM_SEEDS:
            tf.random.set_seed(seed); np.random.seed(seed)
            print(f"  Seed {seed}:")
            X_tr, y_tr, X_te, y_te, _ = prepare_data(
                reduced, labels, PATCH_SIZE, train_ratio=ratio, seed=seed)
            model = build_full_model(X_tr.shape[1:], y_tr.shape[1])
            m = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=30)
            seed_results.append({'acc': m['accuracy'], 'f1': m['f1'], 'kappa': m['kappa']})
            # Save best model per ratio
            if seed == 42:
                save_model(model, f"{dataset_name}_ratio{int(ratio*100)}", m)
            del model; tf.keras.backend.clear_session()

        df = pd.DataFrame(seed_results)
        row = {
            'ratio': f"{ratio*100:.0f}%",
            'OA_mean': df['acc'].mean(), 'OA_std': df['acc'].std(),
            'F1_mean': df['f1'].mean(), 'F1_std': df['f1'].std(),
            'Kappa_mean': df['kappa'].mean(), 'Kappa_std': df['kappa'].std(),
        }
        all_results.append(row)
        print(f"  => OA={row['OA_mean']:.4f}±{row['OA_std']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/{dataset_name}_low_label.csv", index=False)
    print(f"\n{results_df.to_string(index=False)}")
    return results_df


def run_ablation_study(data_path, gt_path, dataset_name="indian_pines"):
    print(f"\n{'='*60}\n  ABLATION STUDY: {dataset_name}\n{'='*60}")

    data, labels = load_dataset(dataset_name, data_path, gt_path)
    preprocessed = preprocess_data(data)
    reduced = apply_pca(preprocessed, PCA_COMPONENTS)
    X_tr, y_tr, X_te, y_te, _ = prepare_data(
        reduced, labels, PATCH_SIZE, train_ratio=0.80, seed=42)
    input_shape = X_tr.shape[1:]
    num_classes = y_tr.shape[1]

    configs = [
        ("No Ensemble (GAP only)", {"use_ensemble": False}),
        ("No Transformer",         {"use_transformer": False}),
        ("No U-Net",               {"use_unet": False}),
        ("Single-scale Conv",      {"use_multiscale": False}),
        ("1 Trans Layer",          {"num_trans_layers": 1}),
        ("2 Trans Layers",         {"num_trans_layers": 2}),
        ("4 Trans Layers",         {"num_trans_layers": 4}),
        ("2 Attn Heads",           {"heads": 2}),
        ("8 Attn Heads",           {"heads": 8}),
        ("Spectral Dim 16",        {"spectral_channels": 16}),
        ("Spectral Dim 64",        {"spectral_channels": 64}),
        ("Token Dim 32",           {"dim": 32}),
        ("Token Dim 128",          {"dim": 128}),
        ("Dropout 0.1",            {"dropout": 0.1}),
        ("Dropout 0.5",            {"dropout": 0.5}),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\n── {name} ──")
        tf.keras.backend.clear_session()
        tf.random.set_seed(42); np.random.seed(42)
        model = build_full_model(input_shape, num_classes, **kwargs)
        m = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=10)
        results.append({
            'Configuration': name,
            'OA (%)': round(m['accuracy']*100, 2),
            'F1 (%)': round(m['f1']*100, 2),
            'Kappa': round(m['kappa'], 4),
            'Params (M)': round(model.count_params()/1e6, 2),
            'Train (s)': round(m['train_time_s'], 1),
        })
        del model

    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/{dataset_name}_ablation.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    return df


def run_statistical_significance(data_path, gt_path, dataset_name="indian_pines"):
    print(f"\n{'='*60}\n  STATISTICAL SIGNIFICANCE: {dataset_name}\n{'='*60}")

    data, labels = load_dataset(dataset_name, data_path, gt_path)
    preprocessed = preprocess_data(data)
    reduced = apply_pca(preprocessed, PCA_COMPONENTS)

    def run_seeds(builder_fn, label):
        accs = []
        for seed in RANDOM_SEEDS:
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed); np.random.seed(seed)
            X_tr, y_tr, X_te, y_te, _ = prepare_data(
                reduced, labels, PATCH_SIZE, train_ratio=0.80, seed=seed)
            model = builder_fn(X_tr.shape[1:], y_tr.shape[1])
            m = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=10)
            accs.append(m['accuracy']); del model
        print(f"  {label}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        return accs

    full_accs = run_seeds(lambda s, c: build_full_model(s, c), "Full Model")
    base_accs = run_seeds(
        lambda s, c: build_full_model(s, c, use_transformer=False), "No Transformer")

    t_stat, p_val = stats.ttest_rel(full_accs, base_accs)
    print(f"\n  Paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    sig = "YES (p<0.05)" if p_val < 0.05 else "NO (p>=0.05)"
    print(f"  Significant: {sig}")
    return {'full': full_accs, 'base': base_accs, 't': t_stat, 'p': p_val}


if __name__ == "__main__":
    # ── SET YOUR PATHS ──
    IP_DATA = "indianpines/indianpinearray.npy"
    IP_GT   = "indianpines/IPgt.npy"

    # Change these for local:
    PU_DATA = "pavia/PaviaU.mat"
    PU_GT   = "pavia/PaviaU_gt.mat"

    # 1. Hyperparameter table (instant)
    hp_df = run_hyperparameter_table()
 
    # 2. Main experiment — trains once, saves model
    model, metrics, comp, noise_df, auc_scores = run_main_experiment(IP_DATA, IP_GT)

    # 3. Low-label experiments
    low_label_df = run_low_label_experiments(IP_DATA, IP_GT)

    # 4. Ablation study
    ablation_df = run_ablation_study(IP_DATA, IP_GT)

    # 5. Statistical significance
    sig_results = run_statistical_significance(IP_DATA, IP_GT)

    # ── Later: reload without retraining ──
    model, metrics = load_saved_model("indian_pines_main")
    print(f"Loaded model accuracy: {metrics.get('accuracy')}")

    if PU_DATA:
        run_main_experiment(PU_DATA, PU_GT,"pavia_university")

    print(f"\n{'='*60}")
    print(f"  ALL DONE — Results in: {RESULTS_DIR}/")
    print(f"  Models in: {MODELS_DIR}/")
    print(f"{'='*60}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {RESULTS_DIR}/{f}")
    for f in sorted(os.listdir(MODELS_DIR)):
        print(f"  {MODELS_DIR}/{f}")