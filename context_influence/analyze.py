import umap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


def compute_distances(no_ctx, ctx):
    num_layers = no_ctx.shape[1]

    distances = []
    for layer in range(num_layers):
        d = cosine_distances(no_ctx[:, layer, :], ctx[:, layer, :])
        distances.append(float(np.mean(np.diag(d))))

    return distances


def compute_embedding(no_ctx, ctx, task_labels, layer_idx, method="pca", dim=2):
    X = np.vstack([no_ctx[:, layer_idx, :], ctx[:, layer_idx, :]])

    labels_ctx = ["no_ctx"] * len(no_ctx) + ["ctx"] * len(ctx)
    labels_task = task_labels + task_labels

    if method == "pca":
        model = PCA(n_components=dim)
        X_emb = model.fit_transform(X)
        extra = {"explained_variance": model.explained_variance_ratio_.tolist()}

    elif method == "umap":
        model = umap.UMAP(
            n_components=dim,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        X_emb = model.fit_transform(X)
        extra = {}

    else:
        raise ValueError("method must be 'pca' or 'umap'")

    points = []
    for i in range(len(X_emb)):
        p = {
            "context": labels_ctx[i],
            "task": labels_task[i],
            "index": int(i % len(task_labels)),
        }

        if dim == 2:
            p["x"], p["y"] = float(X_emb[i, 0]), float(X_emb[i, 1])
        elif dim == 3:
            p["x"], p["y"], p["z"] = map(float, X_emb[i])

        points.append(p)

    return {
        "layer": int(layer_idx),
        "method": method,
        "dim": dim,
        "points": points,
        **extra,
    }


def compute_multi_embedding(
    no_ctx, ctx, task_labels, layer_indices, method="pca", dim=2
):
    return [
        compute_embedding(
            no_ctx,
            ctx,
            task_labels,
            layer_idx,
            method=method,
            dim=dim,
        )
        for layer_idx in layer_indices
    ]


def drop_points_from_result(pca):
    return {
        "layer": pca["layer"],
        "explained_variance": pca.get("explained_variance"),
    }


def drop_points_from_multi_result(pca_layers):
    return [
        {
            "layer": layer["layer"],
            "explained_variance": layer.get("explained_variance"),
        }
        for layer in pca_layers
    ]
