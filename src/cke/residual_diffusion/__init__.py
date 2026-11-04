"""Layer 4 (optional): small residual diffusion (IPSL-AID inspired). STUB, disabled by default.
Only enable if the kernel model leaves large structured residual error."""


def fit_residual(*args, **kwargs):
    raise NotImplementedError("Residual diffusion is optional and not implemented (enable only if needed)")
