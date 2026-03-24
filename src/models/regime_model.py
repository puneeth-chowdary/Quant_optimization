from hmmlearn.hmm import GaussianHMM
import numpy as np

def build_hmm_model(n_components=4):
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=500,
        min_covar=1e-4,
        init_params="stc",   # keep provided means
        random_state=42
    )
    model.startprob_ = np.full(n_components, 1.0 / n_components)
    model.transmat_  = np.full((n_components, n_components), 1.0 / n_components)

    return model