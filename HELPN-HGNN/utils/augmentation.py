import torch


def drop_edge(data, drop_prob=0.2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges, device=data.edge_index.device) > drop_prob
        data_aug = data.clone()
        data_aug.edge_index = data.edge_index[:, mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data_aug.edge_attr = data.edge_attr[mask]
        return data_aug
    return data


def add_noise_to_features(data, noise_std=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if hasattr(data, 'x') and data.x is not None:
        noise = torch.randn_like(data.x) * noise_std
        data_aug = data.clone()
        data_aug.x = data.x + noise
        return data_aug
    return data


def augment_graph(data, drop_prob=0.2, noise_std=0.1, seed=None):
    if drop_prob > 0:
        data = drop_edge(data, drop_prob=drop_prob, seed=seed)
    if noise_std > 0:
        data = add_noise_to_features(data, noise_std=noise_std, seed=seed)
    return data