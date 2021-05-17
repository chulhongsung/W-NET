# source: https://github.com/fkodom/wnet-unsupervised-image-segmentation

import numpy as np
# pip install git+https://github.com/lucasb-eyer/pydensecrf.git
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def crf_fit_predict(softmax: np.ndarray, image: np.ndarray, niter: int = 150):
    unary = unary_from_softmax(softmax).reshape(softmax.shape[0], -1)
    bilateral = create_pairwise_bilateral(sdims=(25, 25), schan=(0.05, 0.05), img=image, chdim=0)

    crf = dcrf.DenseCRF2D(image.shape[2], image.shape[1], softmax.shape[0])
    crf.setUnaryEnergy(unary)
    crf.addPairwiseEnergy(bilateral, compat=100)
    pred = crf.inference(niter)

    return np.array(pred).reshape((-1, image.shape[1], image.shape[2]))

def crf_batch_fit_predict(probabilities: np.ndarray, images: np.ndarray, niter: int = 150):
    
    return np.stack([crf_fit_predict(p, x, niter) for p, x in zip(probabilities, images)], 0)
