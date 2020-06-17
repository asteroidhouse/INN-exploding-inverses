import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
import ipdb
import copy

from scipy import linalg
class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

def compute_is_from_preds(preds, splits):
    # Now compute the mean kl-div
    N = len(preds)
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return split_scores

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1, return_preds=False):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    # ipdb.set_trace()
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    if isinstance(imgs, torch.Tensor):
        imgs = IgnoreLabelDataset(torch.utils.data.TensorDataset(imgs))
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    # ipdb.set_trace()
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy(), x.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    acts = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i], acts[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # ipdb.set_trace()

    opreds = copy.deepcopy(preds)
    np.random.shuffle(preds)

    split_scores = compute_is_from_preds(preds, splits)
    ret_val = [np.mean(split_scores), np.std(split_scores)]
    if return_preds:
        ret_val += [opreds, acts]
    return ret_val




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def run_fid(data, sample):
    assert data.max() <=1 and  data.min() >= 0
    assert sample.max() <=1 and  sample.min() >= 0
    data = 2*data - 1
    if data.shape[1] == 1:
        data = data.repeat(1,3,1,1)
    data = data.detach()
    with torch.no_grad():
        iss, _, _, acts_real = inception_score(data, cuda=True, batch_size=32, resize=True, splits=10, return_preds=True)
    sample = 2*sample - 1
    if sample.shape[1] == 1:
        sample = sample.repeat(1,3,1,1)
    sample = sample.detach()

    with torch.no_grad():
        issf, _, _, acts_fake = inception_score(sample, cuda=True, batch_size=32, resize=True, splits=10, return_preds=True)
    # idxs_ = np.argsort(np.abs(acts_fake).sum(-1))[:1800] # filter the ones with super large values
    # acts_fake = acts_fake[idxs_]
    m1, s1 = calculate_activation_statistics(acts_real)
    m2, s2 = calculate_activation_statistics(acts_fake)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
