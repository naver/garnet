from typing import Tuple, List, Optional, Union

import torch
import torch.nn.functional as F


def calc_PSNR(y,yhat):
    batch,_,_,_ = y.shape
    mse = torch.mean( ((y - yhat) ** 2).view(batch,-1),dim=1)
    # when MSE = 0, PSNR has invinite value
    # To ignore divided by zero, we add 1e-10 to mse.
    # Note that PSNR between same image is arround 140
    return 20 * torch.log10(255.0 / torch.sqrt(mse+ 1e-10))

def calc_AGE(y,yhat):
    # y and yhat are gray scale images
    batch = y.size(0)
    return torch.mean( torch.abs(y - yhat).view(batch,-1),dim=1)

## ssim reference https://github.com/photosynthesis-team/piq/blob/master/piq/ssim.py
## To use PyTorch 1.2.0, we take code of ssim, not install piq.
"""
@misc{piq,
  title={{PyTorch Image Quality}: Metrics and Measure for Image Quality Assessment},
  url={https://github.com/photosynthesis-team/piq},
  note={Open-source software available at https://github.com/photosynthesis-team/piq},
  author={Sergey Kastryulin and Dzhamil Zakirov and Denis Prokopenko},
  year={2019},
}
"""

def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0., -1.),
    # size_dim_range: Tuple[float, float] = (0., -1.),
    size_range: Optional[Tuple[int, int]] = None,
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'

        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]: size_range[1]] == x.size()[size_range[0]: size_range[1]], \
                f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], \
                f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), \
                f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], \
                f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.
    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Uknown reduction. Expected one of {'none', 'mean', 'sum'}")


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         downsample: bool = True, k1: float = 0.01, k2: float = 0.03) -> List[torch.Tensor]:
    r"""Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.
    To match performance with skimage and tensorflow set ``'downsample' = True``.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        full: Return cs map or not.
        downsample: Perform average pool before SSIM computation. Default: True
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.
    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x.type(torch.float32)
    y = y.type(torch.float32)

    x = x / data_range
    y = y / data_range

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    return ssim_val

def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs
