import torch


# ------------- Metrics -------------


def mae(y, y_prim):
    """
    Mean absolute error

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Tensor
    """
    return torch.mean(torch.abs(y - y_prim))


def mse(y, y_prim):
    """
    Mean square error

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Tensor
    """
    return torch.mean((y - y_prim) ** 2)


def r2score(y, y_prim):
    """
    Coefficient of determination (R2 score)

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Float
    """
    return 1 - (torch.sum((y - y_prim) ** 2) / (torch.sum((y - torch.mean(y_prim)) ** 2)))


def relative_r2score(x, y, y_prim):
    """
    Coefficient of determination (R2 score)

    :param x: FloatTensor
    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Float
    """
    return min(1.0, max(1e-8, r2score(y, y_prim)) ** 2 / max(1e-8, r2score(y, x)))


def huber(y, y_prim, g: float):
    """
    Huber loss

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :param g: float
    :return: Tensor
    """
    if float(torch.mean(torch.abs(y - y_prim))) <= g:
        return torch.mean(((y - y_prim) ** 2) / 2)
    else:
        return torch.mean(g * (torch.abs(y - y_prim)) - (g ** 2) / 2)


def rmse(y, y_prim):
    """
    Root mean square error

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Tensor
    """
    return torch.sqrt(mse(y, y_prim))


def weighted_rmse(y, y_prim, lat):
    """
    Root mean square error

    :param lat: List
    :param y: FloatTensor
    :param y_prim: FloatTensor
    :return: Tensor
    """
    weighted_lat = torch.cos(torch.deg2rad(lat))
    weighted_lat /= torch.mean(weighted_lat)

    return torch.sqrt(torch.mean(((y - y_prim) ** 2) * weighted_lat))


def nrmse(y, y_prim, norm: bool, data_min: float, data_max: float):
    """
    Normalised root mean square error

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :param norm: bool
    :param data_min: float
    :param data_max: float
    :return: Tensor
    """
    if norm:
        y = (y - data_min) / (data_max - data_min)
        y_prim = (y_prim - data_min) / (data_max - data_min)

    return rmse(y, y_prim)


def kl(y, y_prim, norm: bool, data_min: float, data_max: float, eps: float):
    """
    Kullback–Leibler divergence

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :param norm: bool
    :param data_min: float
    :param data_max: float
    :param eps: float
    :return: FloatTensor
    """
    if norm:
        y = (y - data_min) / (data_max - data_min)
        y_prim = (y_prim - data_min) / (data_max - data_min)

    return torch.mean(y_prim * torch.log((y_prim + eps) / (y + eps)) - y_prim + y)


def focalKL(y, y_prim, norm: bool, data_min: float, data_max: float, gamma: float, eps: float):
    """
    Focal loss

    :param y: FloatTensor
    :param y_prim: FloatTensor
    :param norm: bool
    :param data_min: float
    :param data_max: float
    :param gamma: float
    :param eps: float
    :return: FloatTensor
    """
    if norm:
        y = (y - data_min) / (data_max - data_min)
        y_prim = (y_prim - data_min) / (data_max - data_min)

    return torch.mean(((1 - y_prim) * gamma) * (torch.log(((y_prim + eps) / (y + eps))) - (1 - y_prim) * gamma) + y)
