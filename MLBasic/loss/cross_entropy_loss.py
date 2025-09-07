import numpy as np

def cross_entropy_loss(labels, preds):
    """
    calculate cross_entropy_loss (log loss for binary classification)
      loss = -labels * log(preds) - (1 - labels) * log(1 - preds)
    """

    if len(labels) != len(preds):
        raise ValueError("labels num should equal to the preds num")

    z = np.array(labels)
    x = np.array(preds)
    res = -z * np.log(x) - (1 - z) * np.log(1 - x)
    return res.tolist()