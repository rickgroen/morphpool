import torch
import morphpool_cuda


class BaselinePool2D(torch.nn.Module):

    def __init__(self, kernel_size=3, stride=2):
        super(BaselinePool2D, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        # Make sure to pad to achieve the following:
        # - With stride == 1, input_shape = output_shape
        # - With stride == 2, output_shape = ceil(input_shape / 2)
        if self.ks % 2 == 0:
            self.pad_with = (0, self.ks // 2, 0, self.ks // 2)
        else:
            self.pad_with = (self.ks // 2, self.ks // 2, self.ks // 2, self.ks // 2)

    def forward(self, xs: torch.Tensor) -> tuple:
        padded_xs = torch.nn.functional.pad(xs, self.pad_with, value=-10.)
        pooled_xs, ind = self.pool(padded_xs)
        return pooled_xs, ind


class BaselineParameterizedPool2D(BaselinePool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero'):
        super(BaselineParameterizedPool2D, self).__init__(kernel_size, stride)
        self.in_channels = in_channels
        h = torch.empty((1, kernel_size ** 2, in_channels))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def _pad(self, xs):
        return torch.nn.functional.pad(xs, self.pad_with, value=-10.)

    def forward(self, xs: torch.Tensor) -> tuple:
        _, _, original_h, original_w = xs.shape
        padded_xs = self._pad(xs)
        b, _, h, w = padded_xs.shape
        pooled_xs = torch.empty(xs.shape, dtype=torch.float32, device=xs.device)
        provenances_xs = torch.empty(xs.shape, dtype=torch.int64, device=xs.device)
        for c in range(self.in_channels):
            unfolded_channel = torch.nn.Unfold(self.ks)(padded_xs[:, c, :, :].view(b, 1, h, w))
            added_channel = unfolded_channel + self.h[:, :, c].view(1, self.ks ** 2, 1)
            maxes, provenances = torch.max(added_channel, dim=1)
            maxes, provenances = maxes.view(b, original_h, original_w), provenances.view(b, original_h, original_w)
            pooled_xs[:, c, :, :], provenances_xs[:, c, :, :] = maxes, provenances
        return pooled_xs[:, :, ::self.stride, ::self.stride], provenances_xs[:, :, ::self.stride, ::self.stride]


class CudaPool2D(torch.nn.Module):

    def __init__(self, kernel_size: int, stride: int = 2, device: int = 0) -> None:
        super(CudaPool2D, self).__init__()
        self.ks = kernel_size
        self.stride = stride
        self.device = device

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return Pool2DAutogradFunction.apply(f, self.ks, self.stride, self.device)

    def extra_repr(self) -> str:
        return f'kernel_size={self.ks}, stride={self.stride}'


class Pool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, kernel_size, stride, device):
        # Perform a forward pass.
        outputs, provenance = morphpool_cuda.maxpool_forward(input, kernel_size, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = kernel_size, stride, input.shape[2], input.shape[3], device
        return outputs, provenance

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor, delta_provenance: torch.Tensor) -> tuple:
        provenance, kernel_size, stride, h, w, device = *ctx.saved_tensors, ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        dldf = morphpool_cuda.maxpool_backward(delta_up, provenance, kernel_size, stride, h, w, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None


class CudaParameterizedPool2D(CudaPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(CudaParameterizedPool2D, self).__init__(kernel_size, stride)
        self.in_channels = in_channels
        h = torch.empty((in_channels, kernel_size, kernel_size))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)
        self.device = device

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return ParameterizedPool2DAutogradFunction.apply(f, self.h, self.stride, self.device)


class ParameterizedPool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, h, stride, device):
        # Perform a forward pass.
        outputs, provenance = morphpool_cuda.parameterized_maxpool_forward(input, h, stride, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(provenance)
        ctx.save_for_backward(provenance)
        ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device = h.shape[-1], stride, input.shape[2], input.shape[3], device
        return outputs, provenance

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor, delta_provenance: torch.Tensor) -> tuple:
        provenance, kernel_size, stride, h, w, device = *ctx.saved_tensors, ctx.kernel_size, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        dldf = morphpool_cuda.parameterized_maxpool_backward_f(delta_up, provenance, kernel_size, stride, h, w, device)
        dldh = morphpool_cuda.parameterized_maxpool_backward_h(delta_up, provenance, kernel_size, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. stride and return indices bool.
        return dldf, dldh, None, None


class CudaParabolicPool2D(CudaPool2D):

    def __init__(self, in_channels, kernel_size=3, stride=2, init='zero', device: int = 0):
        super(CudaParabolicPool2D, self).__init__(kernel_size, stride)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels, ))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)
        self.device = device

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z_i = torch.linspace(-self.ks // 2 + 1, self.ks // 2, self.ks, dtype=torch.float32).to(self.device)
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, suhc that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        h = - z * self.t.view(-1, 1, 1)
        # And return the standard CUDA dilation.
        return ParameterizedPool2DAutogradFunction.apply(f, h, self.stride, self.device)
