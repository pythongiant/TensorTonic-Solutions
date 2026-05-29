import numpy as np
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    def q(i):
        if i == target:
            q_i = (1-epsilon) + epsilon/len(predictions)
        else:
            q_i = epsilon/len(predictions)

        return q_i
        
    sum = 0
    for i in range(0, len(predictions)):
        sum += q(i) * np.log(predictions[i])

    return -sum