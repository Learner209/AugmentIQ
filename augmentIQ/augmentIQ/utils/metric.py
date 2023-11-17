import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

from torchmetrics import SpearmanCorrCoef
from torchmetrics.regression import PearsonCorrCoef


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred)**2)) / np.sqrt(np.sum((true - true.mean())**2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true)**2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def compute_srcc(preds, targets):
    srcc, _ = spearmanr(preds, targets)
    return srcc


def compute_plcc(preds, targets):
    plcc, _ = pearsonr(preds, targets)
    return plcc


def compute_krcc(preds, targets):
    krcc, _ = kendalltau(preds, targets)
    return krcc


def metric(pred, true):
    spcc_metric = SpearmanCorrCoef().to(pred.device)
    plcc_metric = PearsonCorrCoef().to(pred.device)

    spcc = spcc_metric(pred, true)
    plcc = plcc_metric(pred, true)
    return spcc.detach().cpu().numpy(), plcc.detach().cpu().numpy()


if __name__ == '__main__':
    pred = np.array([1, 2, 3, 4, 5])
    true = np.array([1, 2, 3, 4, 3])
    print(metric(pred, true))
    print(compute_srcc(pred, true))
    print(compute_plcc(pred, true))
    print(compute_krcc(pred, true))
