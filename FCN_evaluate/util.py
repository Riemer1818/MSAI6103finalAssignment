# The following code is modified from https://github.com/shelhamer/clockwork-fcn
import numpy as np
import torch


def get_out_scoremap(output):
    return output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def segrun(net, in_, device):
    if isinstance(in_, np.ndarray):
        in_tensor = torch.from_numpy(in_).unsqueeze(0).to(device)
    else:
        in_tensor = in_.unsqueeze(0).to(device)  # Assuming in_ is already a Tensor
    with torch.no_grad():
        output = net(in_tensor)
    # Retrieve the output score map
    return get_out_scoremap(output)


def fast_hist(a, b, n):
    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)


def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu
