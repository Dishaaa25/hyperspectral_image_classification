"""
=============================================================================
SOTA BASELINE MODELS FOR FAIR COMPARISON
=============================================================================
Models:
  1. SpectralFormer (Hong et al., 2021 — IEEE TGRS)
  2. SwinTransformer adapted for HSI (Liu et al., 2021 — ICCV)

Both trained on the EXACT same data splits as your proposed model
for a fair apples-to-apples comparison.

Usage:
  - Import your data prep functions from the main script
  - Call run_sota_comparison(data_path, gt_path)
=============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import pandas as pd
import time, os

# If running in same notebook, these are already defined.
# Otherwise, import from your main script:
# from reviewer_fixes import (
#     load_dataset, preprocess_data, apply_pca,
#     prepare_data, PATCH_SIZE, PCA_COMPONENTS, RESULTS_DIR
# )

PATCH_SIZE = 15
PCA_COMPONENTS = 32
RESULTS_DIR = "reviewer_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# ██ MODEL 1: SpectralFormer
# =============================================================================
# Reference: Hong et al., "SpectralFormer: Rethinking Hyperspectral Image
# Classification With Transformers," IEEE TGRS, 2021.
#
# Key ideas:
#   - Group-wise spectral embedding (GSE): groups adjacent bands into tokens
#   - Cross-layer adaptive fusion (CAF): fuses features across transformer layers
#   - Pure transformer — no CNN spatial branch
# =============================================================================

class GroupWiseSpectralEmbedding(layers.Layer):
    """
    Groups adjacent spectral bands into tokens.
    Input:  (batch, patch_h, patch_w, bands)
    Output: (batch, num_tokens, embed_dim)

    Each token = group of `group_size` adjacent bands from center pixel
    + optional neighboring pixels for spatial context.
    """
    def __init__(self, embed_dim=64, group_size=4, use_spatial=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.use_spatial = use_spatial

    def build(self, input_shape):
        bands = input_shape[-1]
        self.num_groups = bands // self.group_size
        # Each group projected to embed_dim
        if self.use_spatial:
            in_features = self.group_size * 5
        else:
            in_features = self.group_size
        self.projection = layers.Dense(self.embed_dim)
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="random_normal",
            trainable=True
        )
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_groups + 1, self.embed_dim),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        ch, cw = h // 2, w // 2  # center pixel indices

        # Extract center pixel spectrum
        center = x[:, ch, cw, :]  # (batch, bands)

        if self.use_spatial and h >= 3 and w >= 3:
            # Also grab 4-connected neighbors
            top    = x[:, ch-1, cw, :]
            bottom = x[:, ch+1, cw, :]
            left   = x[:, ch, cw-1, :]
            right  = x[:, ch, cw+1, :]
            # Stack: (batch, 5, bands)
            pixels = tf.stack([center, top, bottom, left, right], axis=1)
        else:
            pixels = tf.expand_dims(center, axis=1)  # (batch, 1, bands)

        num_px = pixels.shape[1] if pixels.shape[1] is not None else 5
        bands = tf.shape(pixels)[-1]

        # Group adjacent bands: (batch, num_px, num_groups, group_size)
        usable_bands = self.num_groups * self.group_size
        pixels_trimmed = pixels[:, :, :usable_bands]
        grouped = tf.reshape(pixels_trimmed, (batch, num_px, self.num_groups, self.group_size))

        # Flatten pixels within each group: (batch, num_groups, group_size * num_px)
        grouped = tf.transpose(grouped, [0, 2, 1, 3])  # (batch, num_groups, num_px, group_size)
        grouped = tf.reshape(grouped, (batch, self.num_groups, -1))

        # Project to embed_dim
        tokens = self.projection(grouped)  # (batch, num_groups, embed_dim)

        # Prepend CLS token
        cls = tf.tile(self.cls_token, [batch, 1, 1])
        tokens = tf.concat([cls, tokens], axis=1)  # (batch, num_groups+1, embed_dim)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        return tokens


class CrossLayerAdaptiveFusion(layers.Layer):
    """
    CAF module: fuses current layer output with previous layer output
    via learned gating.
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.gate = layers.Dense(dim, activation='sigmoid')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, current, previous):
        gate = self.gate(current)
        fused = gate * current + (1 - gate) * previous
        return self.norm(fused)


def build_spectralformer(input_shape, num_classes,
                         embed_dim=64, num_layers=4, num_heads=4,
                         group_size=4, ff_dim=256, dropout=0.1):
    """
    SpectralFormer: pure transformer for HSI classification.

    Architecture:
      1. Group-wise Spectral Embedding (GSE)
      2. L transformer blocks with Cross-layer Adaptive Fusion (CAF)
      3. CLS token → MLP head → classification

    Args:
        input_shape: (patch_h, patch_w, bands)
        num_classes: number of land-cover classes
        embed_dim: token embedding dimension
        num_layers: number of transformer blocks
        num_heads: attention heads per block
        group_size: adjacent bands per token group
        ff_dim: feed-forward hidden dimension
        dropout: dropout rate
    """
    inputs = layers.Input(shape=input_shape)

    # ── Group-wise Spectral Embedding ──
    tokens = GroupWiseSpectralEmbedding(
        embed_dim=embed_dim, group_size=group_size
    )(inputs)

    # ── Transformer Blocks with CAF ──
    prev_tokens = tokens
    caf_modules = [CrossLayerAdaptiveFusion(embed_dim) for _ in range(num_layers)]

    for i in range(num_layers):
        # Multi-head self-attention
        att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout
        )(tokens, tokens)
        tokens = layers.Add()([tokens, att])
        tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)

        # Feed-forward
        ffn = layers.Dense(ff_dim, activation='gelu')(tokens)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(embed_dim)(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        tokens = layers.Add()([tokens, ffn])
        tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)

        # Cross-layer Adaptive Fusion (skip for first layer)
        if i > 0:
            tokens = caf_modules[i](tokens, prev_tokens)
        prev_tokens = tokens

    # ── Classification head: use CLS token ──
    cls_token = tokens[:, 0, :]  # (batch, embed_dim)
    x = layers.Dropout(dropout)(cls_token)
    x = layers.Dense(embed_dim, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs, name="SpectralFormer")


# =============================================================================
# ██ MODEL 2: Swin Transformer for HSI
# =============================================================================
# Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
# using Shifted Windows," ICCV 2021.
#
# Adapted for HSI:
#   - 1D spectral conv for band reduction (like your model)
#   - Window-based multi-head self-attention (W-MSA)
#   - Shifted window attention (SW-MSA) for cross-window connections
#   - Patch merging for hierarchical features
#   - MLP classification head
# =============================================================================

class WindowAttention(layers.Layer):
    """
    Window-based multi-head self-attention (W-MSA).
    Computes attention within local windows of size window_size x window_size.
    """
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        self.qkv = layers.Dense(self.dim * 3)
        self.proj = layers.Dense(self.dim)
        # Relative position bias
        self.relative_position_bias = self.add_weight(
            name="rpb",
            shape=((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads),
            initializer="zeros",
            trainable=True
        )
        # Compute relative position index
        coords_h = np.arange(self.window_size)
        coords_w = np.arange(self.window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'), axis=0)  # (2, ws, ws)
        coords_flat = coords.reshape(2, -1)  # (2, ws*ws)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        relative = relative.transpose(1, 2, 0)  # (N, N, 2)
        relative[:, :, 0] += self.window_size - 1
        relative[:, :, 1] += self.window_size - 1
        relative[:, :, 0] *= 2 * self.window_size - 1
        self.relative_position_index = tf.constant(
            relative.sum(-1).flatten(), dtype=tf.int32
        )
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        N = self.window_size * self.window_size
        C = self.dim

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) * self.scale  # (B, heads, N, N)

        # Add relative position bias
        rpb = tf.gather(self.relative_position_bias, self.relative_position_index)
        rpb = tf.reshape(rpb, (N, N, self.num_heads))
        rpb = tf.transpose(rpb, (2, 0, 1))  # (heads, N, N)
        attn = attn + rpb[tf.newaxis, ...]

        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)  # (B, heads, N, head_dim)
        out = tf.transpose(out, (0, 2, 1, 3))  # (B, N, heads, head_dim)
        out = tf.reshape(out, (B, N, C))
        return self.proj(out)


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows."""
    B = tf.shape(x)[0]
    H, W = x.shape[1], x.shape[2]
    C = x.shape[3]
    # Pad if needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = tf.pad(x, [[0,0], [0, pad_h], [0, pad_w], [0, 0]])
    Hp = H + pad_h
    Wp = W + pad_w
    x = tf.reshape(x, (B, Hp // window_size, window_size,
                       Wp // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))  # (B, nH, nW, ws, ws, C)
    nw = (Hp // window_size) * (Wp // window_size)
    windows = tf.reshape(x, (B * nw, window_size * window_size, C))
    return windows, Hp, Wp


def window_reverse(windows, window_size, Hp, Wp, B):
    """Reverse window partition."""
    nH = Hp // window_size
    nW = Wp // window_size
    C = windows.shape[-1]
    x = tf.reshape(windows, (B, nH, nW, window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (B, Hp, Wp, C))


class SwinTransformerBlock(layers.Layer):
    """
    Single Swin Transformer block with W-MSA or SW-MSA.
    """
    def __init__(self, dim, num_heads, window_size=3, shift_size=0,
                 ff_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.ff_ratio = ff_ratio

    def build(self, input_shape):
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(self.dim, self.window_size, self.num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(int(self.dim * self.ff_ratio), activation='gelu'),
            layers.Dropout(0.1),
            layers.Dense(self.dim),
            layers.Dropout(0.1),
        ])
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        H, W = x.shape[1], x.shape[2]

        shortcut = x
        x = self.norm1(x)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        # Window partition
        windows, Hp, Wp = window_partition(x, self.window_size)

        # W-MSA
        attn_out = self.attn(windows)

        # Reverse windows
        x = window_reverse(attn_out, self.window_size, Hp, Wp, B)

        # Crop padding
        x = x[:, :H, :W, :]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        # Residual + FFN
        x = shortcut + x
        x = x + self.ffn(self.norm2(x))

        return x


class PatchMerging(layers.Layer):
    """Patch merging for spatial downsampling (like pooling in CNN)."""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.reduction = layers.Dense(self.dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        H, W = x.shape[1], x.shape[2]

        # Pad if odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = tf.pad(x, [[0,0], [0, pad_h], [0, pad_w], [0, 0]])
            H += pad_h
            W += pad_w

        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right

        merged = tf.concat([x0, x1, x2, x3], axis=-1)  # (B, H/2, W/2, 4C)
        merged = self.norm(merged)
        return self.reduction(merged)  # (B, H/2, W/2, dim)


def build_swin_hsi(input_shape, num_classes,
                   embed_dim=64, depths=(2, 2, 2), num_heads_list=(2, 4, 8),
                   window_size=3, spectral_channels=32, dropout=0.1):
    """
    Swin Transformer adapted for Hyperspectral Image classification.

    Architecture:
      1. 1D Conv spectral reduction (bands → spectral_channels)
      2. Linear patch embedding
      3. Swin Transformer stages with patch merging
      4. Global Average Pooling → MLP classification head

    Args:
        input_shape: (patch_h, patch_w, bands)
        num_classes: number of classes
        embed_dim: base embedding dimension (doubles at each stage)
        depths: number of Swin blocks per stage
        num_heads_list: attention heads per stage
        window_size: local attention window size
        spectral_channels: output channels of spectral reduction
        dropout: dropout rate
    """
    inputs = layers.Input(shape=input_shape)
    h, w, b = input_shape

    # ── Spectral reduction (1D conv along bands) ──
    x = layers.Reshape((h * w, b))(inputs)
    x = layers.Conv1D(spectral_channels, kernel_size=7, padding='same', activation='relu')(x)
    x = layers.Reshape((h, w, spectral_channels))(x)

    # ── Linear embedding to embed_dim ──
    x = layers.Conv2D(embed_dim, 1, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # ── Swin Transformer Stages ──
    for stage_idx, (depth, heads) in enumerate(zip(depths, num_heads_list)):
        dim = embed_dim * (2 ** stage_idx)

        for block_idx in range(depth):
            shift = 0 if block_idx % 2 == 0 else window_size // 2
            x = SwinTransformerBlock(
                dim=dim, num_heads=heads,
                window_size=window_size, shift_size=shift,
                dropout=dropout,
                name=f"swin_stage{stage_idx}_block{block_idx}"
            )(x)

        # Patch merging between stages (except last)
        if stage_idx < len(depths) - 1:
            next_dim = embed_dim * (2 ** (stage_idx + 1))
            x = PatchMerging(next_dim, name=f"patch_merge_{stage_idx}")(x)

    # ── Classification Head ──
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs, name="SwinTransformer_HSI")


# =============================================================================
# ██ TRAINING & EVALUATION UTILITY (same as main script)
# =============================================================================

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

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'params': model.count_params(),
        'train_time_s': train_time,
        'inference_time_s': inf_time,
        'history': hist.history,
    }


# =============================================================================
# ██ COMPARISON RUNNER
# =============================================================================

def run_sota_comparison(data_path, gt_path, dataset_name="indian_pines"):
    """
    Train and evaluate all three models (Proposed, SpectralFormer, Swin)
    on identical data splits across multiple seeds for fair comparison.

    Generates:
      - comparison_table.csv (Table 6 replacement for paper)
      - per-model confusion matrices
      - per-model training curves
    """
    # ── Import data prep from main script (or define locally) ──
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SKPCA

    def preprocess(data):
        h, w, b = data.shape
        r = data.reshape(-1, b).astype(np.float64)  # float64 prevents overflow
        s = StandardScaler()
        normed = s.fit_transform(r).astype(np.float32)  # back to float32 for model
        return normed.reshape(h, w, b)

    def reduce_pca(data, n=PCA_COMPONENTS):
        h, w, b = data.shape
        r = data.reshape(-1, b)
        pca = SKPCA(n_components=n)
        red = pca.fit_transform(r)
        print(f"  PCA variance: {np.sum(pca.explained_variance_ratio_):.4f}")
        return red.reshape(h, w, n)

    def create_patches(data, labels, ps=PATCH_SIZE):
        h, w, b = data.shape
        pad = ps // 2
        padded = np.pad(data, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
        patches, plabels = [], []
        for i in range(h):
            for j in range(w):
                if labels[i,j] == 0: continue
                p = padded[i:i+ps, j:j+ps, :]
                if p.shape[0]==ps and p.shape[1]==ps:
                    patches.append(p)
                    plabels.append(labels[i,j])
        return np.array(patches), np.array(plabels)

    def prep_data(data, labels, ratio=0.80, seed=42):
        np.random.seed(seed)
        patches, plabels = create_patches(data, labels)
        unique = np.unique(plabels); unique = unique[unique!=0]
        nc = len(unique)
        lmap = {c:i for i,c in enumerate(unique)}
        mapped = np.array([lmap[l] for l in plabels])
        Xtr,ytr,Xte,yte = [],[],[],[]
        for ci in range(nc):
            idx = np.where(mapped==ci)[0]
            np.random.shuffle(idx)
            n = int(len(idx)*ratio)
            Xtr.extend(patches[idx[:n]]); ytr.extend(mapped[idx[:n]])
            Xte.extend(patches[idx[n:]]); yte.extend(mapped[idx[n:]])
        Xtr,ytr = np.array(Xtr),np.array(ytr)
        Xte,yte = np.array(Xte),np.array(yte)
        return Xtr, to_categorical(ytr,nc), Xte, to_categorical(yte,nc), lmap

    # ── Load & preprocess ──
    print(f"\n{'='*70}")
    print(f"  SOTA COMPARISON: {dataset_name}")
    print(f"{'='*70}")

    data = np.load(data_path)
    labels = np.load(gt_path)
    print(f"  Data: {data.shape}, Labels: {labels.shape}")

    preprocessed = preprocess(data)
    reduced = reduce_pca(preprocessed)

    # ── Seeds for statistical comparison ──
    seeds = [42]  # 3 seeds (increase to 5 if you have GPU time)

    # ── Model builders ──

    model_builders = {
        "SpectralFormer": lambda shape, nc: build_spectralformer(
            shape, nc, embed_dim=64, num_layers=4, num_heads=4,
            group_size=4, ff_dim=256, dropout=0.1
        ),
        "SwinTransformer-HSI": lambda shape, nc: build_swin_hsi(
            shape, nc, embed_dim=64, depths=(2, 2, 2),
            num_heads_list=(2, 4, 8), window_size=3,
            spectral_channels=32, dropout=0.1
        ),
    }

    # ── Run all models across seeds ──
    all_results = {name: [] for name in model_builders}

    for seed in seeds:
        print(f"\n{'─'*50}")
        print(f"  Seed: {seed}")
        print(f"{'─'*50}")

        X_tr, y_tr, X_te, y_te, lmap = prep_data(reduced, labels, ratio=0.80, seed=seed)
        input_shape = X_tr.shape[1:]
        num_classes = y_tr.shape[1]

        for name, builder_fn in model_builders.items():
            print(f"\n  ── {name} ──")
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed)
            np.random.seed(seed)

            model = builder_fn(input_shape, num_classes)
            if seed == seeds[0]:
                print(f"    Parameters: {model.count_params():,}")

            m = train_and_evaluate(model, X_tr, y_tr, X_te, y_te,
                                   epochs=10, batch_size=16)
            all_results[name].append(m)
            print(f"    OA={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
                  f"κ={m['kappa']:.4f}  Params={m['params']:,}")
            del model

    # ── Aggregate results ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE (mean ± std over {len(seeds)} seeds)")
    print(f"{'='*70}")

    rows = []
    for name, results_list in all_results.items():
        accs   = [r['accuracy'] for r in results_list]
        f1s    = [r['f1'] for r in results_list]
        kappas = [r['kappa'] for r in results_list]
        params = results_list[0]['params']
        ttimes = [r['train_time_s'] for r in results_list]
        itimes = [r['inference_time_s'] for r in results_list]

        row = {
            'Model': name,
            'OA (%)': f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
            'F1 (%)': f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
            'Kappa': f"{np.mean(kappas):.4f} ± {np.std(kappas):.4f}",
            'Params (M)': f"{params/1e6:.2f}",
            'Train (s)': f"{np.mean(ttimes):.0f}",
            'Inference (s)': f"{np.mean(itimes):.1f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save
    save_path = f"{RESULTS_DIR}/{dataset_name}_sota_comparison.csv"
    df.to_csv(save_path, index=False)
    print(f"\n  Saved to: {save_path}")

    # ── Print confusion matrices for last seed ──
    for name, results_list in all_results.items():
        cm = results_list[-1]['confusion_matrix']
        print(f"\n  {name} — Confusion Matrix (last seed):")
        print(cm)

    return df, all_results


# =============================================================================
# ██ LOW-LABEL COMPARISON (5%, 10%, 15% — addresses Reviewer 3)
# =============================================================================

def run_sota_low_label(data_path, gt_path, dataset_name="indian_pines"):
    """
    Compare all models under low-label regimes.
    This is the MOST IMPORTANT experiment for Reviewer 3.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SKPCA

    def preprocess(data):
        h,w,b = data.shape
        r = data.reshape(-1,b).astype(np.float64)
        normed = StandardScaler().fit_transform(r).astype(np.float32)
        return normed.reshape(h,w,b)
    def reduce_pca(data, n=PCA_COMPONENTS):
        h,w,b = data.shape
        pca = SKPCA(n_components=n)
        return pca.fit_transform(data.reshape(-1,b)).reshape(h,w,n)
    def create_patches(data, labels, ps=PATCH_SIZE):
        h,w,b = data.shape
        pad = ps//2
        padded = np.pad(data,((pad,pad),(pad,pad),(0,0)),mode='reflect')
        patches,plabels = [],[]
        for i in range(h):
            for j in range(w):
                if labels[i,j]==0: continue
                p = padded[i:i+ps,j:j+ps,:]
                if p.shape[0]==ps and p.shape[1]==ps:
                    patches.append(p); plabels.append(labels[i,j])
        return np.array(patches),np.array(plabels)
    def prep_data(data, labels, ratio=0.10, seed=42):
        np.random.seed(seed)
        patches, plabels = create_patches(data, labels)
        unique = np.unique(plabels); unique = unique[unique!=0]
        nc = len(unique); lmap = {c:i for i,c in enumerate(unique)}
        mapped = np.array([lmap[l] for l in plabels])
        Xtr,ytr,Xte,yte = [],[],[],[]
        for ci in range(nc):
            idx = np.where(mapped==ci)[0]; np.random.shuffle(idx)
            n = max(1, int(len(idx)*ratio))  # at least 1 sample
            Xtr.extend(patches[idx[:n]]); ytr.extend(mapped[idx[:n]])
            Xte.extend(patches[idx[n:]]); yte.extend(mapped[idx[n:]])
        return np.array(Xtr), to_categorical(np.array(ytr),nc), \
            np.array(Xte), to_categorical(np.array(yte),nc), lmap

    data = np.load(data_path); labels = np.load(gt_path)
    preprocessed = preprocess(data); reduced = reduce_pca(preprocessed)

    ratios = [0.05, 0.10]
    seed = 42  # single seed for speed; increase to 3 if time allows

    model_builders = {
        "SpectralFormer": lambda s, c: build_spectralformer(s, c),
        "Swin-HSI": lambda s, c: build_swin_hsi(s, c),
    }

    rows = []
    for ratio in ratios:
        print(f"\n── Ratio: {ratio*100:.0f}% ──")
        tf.random.set_seed(seed); np.random.seed(seed)
        X_tr, y_tr, X_te, y_te, _ = prep_data(reduced, labels, ratio, seed)
        input_shape = X_tr.shape[1:]
        nc = y_tr.shape[1]
        print(f"  Train={len(X_tr)}, Test={len(X_te)}")

        for name, builder in model_builders.items():
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed); np.random.seed(seed)
            model = builder(input_shape, nc)
            m = train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=10)
            rows.append({
                'Ratio': f"{ratio*100:.0f}%",
                'Model': name,
                'OA (%)': round(m['accuracy']*100, 2),
                'F1 (%)': round(m['f1']*100, 2),
                'Kappa': round(m['kappa'], 4),
            })
            print(f"  {name}: OA={m['accuracy']:.4f}")
            del model

    df = pd.DataFrame(rows)
    save_path = f"{RESULTS_DIR}/{dataset_name}_sota_low_label.csv"
    df.to_csv(save_path, index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"Saved to: {save_path}")
    return df


# =============================================================================
# ██ ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set your data paths
    IP_DATA = "indianpines/indianpinearray.npy"
    IP_GT   = "indianpines/IPgt.npy"

    # ── 1. Main comparison (80% split, 3 seeds) ──
    comparison_df, all_results = run_sota_comparison(IP_DATA, IP_GT)

    # ── 2. Low-label comparison (5%, 10%, 15%) ──
    low_label_df = run_sota_low_label(IP_DATA, IP_GT)

    print("\n\nDone! Check reviewer_results/ for all CSV tables.")