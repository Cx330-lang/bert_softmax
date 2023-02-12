# -*- encoding -*-

from sklearn.metrics import roc_auc_score

def mean(item):
    res = sum(item) / len(item) if len(item) > 0 else 0

    return res

def accuracy(pred_y, true_y):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    corr = 0

    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1

    acc = corr / len(pred_y) if len(pred_y) > 0 else 0

    return acc

def binary_auc(pred_y, true_y):
    auc = roc_auc_score(true_y, pred_y)
    return auc

def binary_precision(pred_y, true_y, positive=1):
    corr = 0
    pred_corr = 0

    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0

    return prec

def binary_recall(pred_y, true_y, positive=1):
    corr = 0
    true_corr = 0

    for i in range(len(pred_y)):
        if true_y[i] == 1:
            true_corr +=1

            if pred_y[i] == true_y[i]:
                corr += 1
    rec = corr / true_corr if true_corr > 0 else 0

    return rec

def binary_f_beta(pred_y, true_y, beta, positive=1):
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)

    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0

    return f_b

def multi_precision(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precision = {label: binary_precision(pred_y, true_y, label) for label in labels}

    prec = mean(precision.values())

    return prec

def multi_recall(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = {label:binary_recall(pred_y, true_y, label) for label in labels}

    rec = mean(recalls.values())

    return rec

def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = {label:binary_f_beta(pred_y, true_y, label, beta) for label in labels}

    f_beta = mean(f_betas)

    return f_beta

def get_binary_metrics(pred_y, true_y, f_b=1.0):

    acc = accuracy(pred_y, true_y)
    auc = binary_auc(pred_y, true_y)
    recall = multi_recall(pred_y, true_y)
    precision = multi_precision(pred_y, true_y)
    f_beta = multi_f_beta(pred_y, true_y, f_b)

    return acc, auc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_b=1.0):
    # print(pred_y, true_y)
    # print(type(pred_y), type(true_y))

    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_b)

    return acc, recall, precision, f_beta












