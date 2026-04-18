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


def compute_pca(no_ctx, ctx, task_labels, layer_idx):
    X = np.vstack([no_ctx[:, layer_idx, :], ctx[:, layer_idx, :]])

    # no_ctx and ctx have shape : (num_tasks, num_layers, hidden_dimension)
    labels_ctx = ["no_ctx"] * len(no_ctx) + ["ctx"] * len(ctx)

    # As both the task (with or without context) are the same like math_easy, 
    # code_bugfix, etc
    labels_task = task_labels + task_labels

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    points = []
    for i in range(len(X2)):
        points.append(
            {
                "x": float(X2[i, 0]),
                "y": float(X2[i, 1]),
                "context": labels_ctx[i],
                "task": labels_task[i],
                "index": int(i % len(task_labels)),
            }
        )

    return {
        "layer": int(layer_idx),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "points": points,
    }


def compute_multi_pca(no_ctx, ctx, task_labels, layer_indices):
    return [
        compute_pca(no_ctx, ctx, task_labels, layer_idx)
        for layer_idx in layer_indices
    ]


def drop_points_from_pca(pca):
    return {
        "layer": pca["layer"],
        "explained_variance": pca.get("explained_variance"),
    }


def drop_points_from_multi(pca_layers):
    return [
        {
            "layer": layer["layer"],
            "explained_variance": layer.get("explained_variance"),
        }
        for layer in pca_layers
    ]

