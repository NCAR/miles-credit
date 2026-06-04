import torch

class AMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        elif reduction == "none":
            self.reduction = lambda x: x
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
    def forward(self, target, pred):
        """
        compute AMSE over last two spatial dimensions
        """
        fft_pred = rfft2_no_conj_pairs(pred)
        fft_target = rfft2_no_conj_pairs(target)
        rfft_weights = get_rfft2_weights(*pred.shape[-2:])
        
        psd_pred = psd_2d(fft_pred, rfft_weights)
        psd_target = psd_2d(fft_target, rfft_weights)

        coherence_k = coherence_2d(fft_pred, fft_target, psd_pred, psd_target, rfft_weights)
        amse_k = (
            (torch.sqrt(psd_pred) - torch.sqrt(psd_target)) ** 2
            + 2 * torch.max(torch.stack([psd_pred, psd_target], dim=0), dim=0)[0]
            * (1. - coherence_k)
        )
        amse = amse_k.mean(dim=-1)
        return self.reduction(amse)

def radial_sum_batched_rfft(batched_array):
    """
    compute the radial sum over the last two dimensions which are based on 2D rFFTs
    """

    shape = batched_array.shape

    flattened_batch = batched_array.view(torch.prod(torch.tensor(shape[:-2])), shape[-2], shape[-1])    
    flat_batch_radial_sum = torch.vmap(radial_sum_array_rfft)(flattened_batch)
    batched_radial_sum = flat_batch_radial_sum.view(*shape[:-2], flat_batch_radial_sum.shape[-1])

    return batched_radial_sum


def radial_sum_array_rfft(array):
    """
    compute the radial sum of 2D array derived from a 2D rFFT
    the 0th freq is at 0,0
    """
    H, W = array.shape[-2:]
    
    y = torch.arange(H, device=array.device)
    x = torch.arange(W, device=array.device)
    
    # Grid of radial distances
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2).long()
    
    max_radius = min(H, W)
    mask = r < max_radius

    # Accumulate values and counts per radius bin
    r_flat = r[mask]
    v_flat = array[mask]

    radial_sum = torch.zeros(max_radius, device=array.device, dtype=array.dtype)
    radial_sum = torch.scatter_add(radial_sum, 0, r_flat, v_flat)

    return radial_sum

def get_rfft2_weights(n,m):
    """
    return the weights needed to compute power spectra like sums. n,m are the sizes of the original array (not the fft)
    from rfft2 arrays without redundant conjugate pairs. 
    """
    weights = torch.ones(n//2 + 1, m//2 + 1)

    # if m even then last entry along m dim is nyquist freq
    weights[: n // 2 + n % 2, : m // 2 + m % 2] = 2 # n//2 + n % 2 = n/2 if n even, (n+1)/2 if n odd (include nyquist freq)
    weights[0, 0] = 1 # reset 0th freq to 0

    return weights
    
def psd_2d(rfft_2d, weights):
    """compute the psd from a real fft"""

    return radial_sum_batched_rfft(weights * torch.abs(rfft_2d) ** 2)

def coherence_2d(fft_x, fft_y, psd_x, psd_y, weights):
    ax_ay_conj = radial_sum_batched_rfft(weights * (fft_x * torch.conj(fft_y)).real)
    return ax_ay_conj / torch.sqrt(psd_x * psd_y)

def rfft2_no_conj_pairs(array):
    """
    compute rfft2 over last two dims n x m, and removes conj pairs
    output has shape n//2 + 1, m//2+1
    """
    n = array.shape[-2]
    return torch.fft.rfft2(array)[..., : n//2 + 1, :]


if __name__ == "__main__":
    b, n, m = 2, 5, 4

    pred = torch.randn((b,n,m))
    target = torch.randn((b,n,m))

    amse = AMSELoss()
    loss = amse(target, pred)
    print(loss)