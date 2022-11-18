import torch
import morphpool_cuda


class BaselineMaxUnpool2D(torch.nn.Module):

    def __init__(self, input_kernel_size, stride=2):
        super().__init__()
        self.in_ks = input_kernel_size
        self.unpool = torch.nn.MaxUnpool2d(input_kernel_size, stride=stride)

    def _unpad(self, xs: torch.Tensor) -> torch.Tensor:
        unpad = self.in_ks // 2
        if self.in_ks % 2 == 1:
            return xs[:, :, unpad:-unpad, unpad:-unpad]
        return xs[:, :, :-unpad, :-unpad]

    def _unfold(self, xs: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = xs.shape
        pad = self.out_ks // 2
        padded_xs = torch.nn.functional.pad(xs, (pad, pad, pad, pad), value=-10.)
        return self.unfold_func(padded_xs).view(b, c_in, self.out_ks ** 2, h * w)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        return self._unpad(x_upsampled)


class BaselineUnpool2D(torch.nn.Module):

    def __init__(self, input_kernel_size, output_kernel_size, stride=2):
        super().__init__()
        self.in_ks = input_kernel_size
        self.out_ks = output_kernel_size
        assert self.out_ks % 2 == 1, f"Output kernel size should be odd, is {self.out_ks}"
        self.unpool = torch.nn.MaxUnpool2d(input_kernel_size, stride=stride)
        self.unfold_func = torch.nn.Unfold(output_kernel_size)

    def _unpad(self, xs: torch.Tensor) -> torch.Tensor:
        unpad = self.in_ks // 2
        if self.in_ks % 2 == 1:
            return xs[:, :, unpad:-unpad, unpad:-unpad]
        return xs[:, :, :-unpad, :-unpad]

    def _unfold(self, xs: torch.Tensor) -> torch.Tensor:
        b, c_in, h, w = xs.shape
        pad = self.out_ks // 2
        padded_xs = torch.nn.functional.pad(xs, (pad, pad, pad, pad), value=-10.)
        return self.unfold_func(padded_xs).view(b, c_in, self.out_ks ** 2, h * w)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        x_upsampled = self._unpad(x_upsampled)
        # Now unfold and do an unparameterized dilation.
        desired_shape = x_upsampled.shape
        x_unfolded = self._unfold(x_upsampled)
        up, _ = torch.max(x_unfolded, dim=2)
        return up.view(*desired_shape)


class BaselineParameterizedUnpool2D(BaselineUnpool2D):

    def __init__(self, in_channels, input_kernel_size, output_kernel_size, stride=2, init='zero'):
        super(BaselineParameterizedUnpool2D, self).__init__(input_kernel_size, output_kernel_size, stride=stride)
        self.in_channels = in_channels
        h = torch.empty((1, in_channels, output_kernel_size ** 2, 1))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        h = torch.arange(0, in_channels * output_kernel_size ** 2, dtype=torch.float32).view((1, in_channels, output_kernel_size ** 2, 1))
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)

    def forward(self, x, indices, size=None):
        # If padding was used in forward pooling, we need to provide the sizes to unpooling explicitly.
        if self.in_ks % 2 == 1:
            up_size = (size[0] + self.in_ks - 1, size[1] + self.in_ks - 1)
        # Unpooling into the same size we put in, except for padding.
        else:
            up_size = (size[0] + 1, size[1] + 1)
        x_upsampled = self.unpool(x, indices, up_size)
        x_upsampled = self._unpad(x_upsampled)
        # Now unfold and do an parameterized dilation.
        desired_shape = x_upsampled.shape
        x_unfolded = self._unfold(x_upsampled)
        x_added = x_unfolded + self.h
        up, _ = torch.max(x_added, dim=2)
        return up.view(*desired_shape), x_upsampled


class CudaMaxUnpool2D(torch.nn.Module):
    """
        This is the regular unpooling by max pooling, like the nn.Unpool2D that torch implements,
        but now with my provenance registration.
    """

    def __init__(self, input_kernel_size, stride=2, device: int = 0) -> None:
        super(CudaMaxUnpool2D, self).__init__()
        self.in_ks = input_kernel_size
        self.stride = stride
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        return MaxUnpool2DAutogradFunction.apply(f, provenance, size, self.in_ks, self.stride, self.device)

    def extra_repr(self) -> str:
        return f'input kernel_size={self.in_ks}, stride={self.stride}'


class MaxUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morphpool_cuda.unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance)
        ctx.save_for_backward(pool_provenance)
        ctx.in_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, stride, h, w, device
        return upsampled_ins

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, in_ks, stride, h, w, device = *ctx.saved_tensors, ctx.in_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Compute delta_up it w.r.t. the up-sampling operation.
        dldf = morphpool_cuda.unpool_backward(delta_up.contiguous(), error_provenance, in_ks, stride, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None, None, None


class CudaMorphUnpool2D(torch.nn.Module):
    """
        Unparameterized Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a flat structuring element.
    """

    def __init__(self, input_kernel_size, output_kernel_size, stride=2, device: int = 0) -> None:
        super(CudaMorphUnpool2D, self).__init__()
        self.in_ks = input_kernel_size
        self.out_ks = output_kernel_size
        assert self.out_ks % 2 == 1, f"Output kernel size should be odd, is {self.out_ks}"
        self.stride = stride
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Input the features, the provenances from the previous pool, and the desired size.
        """
        return MorphUnpool2DAutogradFunction.apply(f, provenance, size, self.in_ks, self.out_ks,
                                                   self.stride, self.device)

    def extra_repr(self) -> str:
        return f'input kernel_size={self.in_ks}, output kernel_size={self.out_ks}, stride={self.stride}'


class MorphUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morphpool_cuda.unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morphpool_cuda.maxpool_forward(upsampled_ins, out_ks, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morphpool_cuda.maxpool_backward(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morphpool_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, None, None, None, None, None, None


class CudaParameterizedMorphUnpool2D(CudaMorphUnpool2D):
    """
        Parameterized Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a free structuring element.
    """

    def __init__(self, in_channels, input_kernel_size, output_kernel_size, stride=2, init='zero', device: int = 0):
        super(CudaParameterizedMorphUnpool2D, self).__init__(input_kernel_size, output_kernel_size, stride=stride)
        self.in_channels = in_channels
        h = torch.empty((in_channels, self.out_ks, self.out_ks))
        if init == 'zero':
            torch.nn.init.zeros_(h)
        else:
            torch.nn.init.kaiming_uniform_(h)
        self.h = torch.nn.parameter.Parameter(h, requires_grad=True)
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        return ParameterizedMorphUnpool2DAutogradFunction.apply(f, self.h, provenance, size,
                                                                self.in_ks, self.out_ks, self.stride, self.device)


class ParameterizedMorphUnpool2DAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, pool_provenance, size, in_ks, out_ks, stride, device):
        h, w = size
        # Perform a forward pass, which is an up sampling first. Also save the pooling provenance,
        # to set the derivatives back at the correct locations.
        upsampled_ins = morphpool_cuda.unpool_forward(inputs, pool_provenance, in_ks, stride, h, w, device)
        # Then perform a max pool with the desired out_ks parameters, at stride 1.
        outputs, backward_provenance = morphpool_cuda.parameterized_maxpool_forward(upsampled_ins, weights, 1, device)
        # Save provenances for the backward, but do not require gradients w.r.t. to it.
        ctx.mark_non_differentiable(pool_provenance, backward_provenance)
        ctx.save_for_backward(pool_provenance, backward_provenance)
        ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device = in_ks, out_ks, stride, h, w, device
        return outputs

    @staticmethod
    def backward(ctx, delta_up: torch.Tensor) -> tuple:
        error_provenance, backward_provenance = ctx.saved_tensors
        in_ks, out_ks, stride, h, w, device = ctx.in_ks, ctx.out_ks, ctx.stride, ctx.h, ctx.w, ctx.device
        # Use the back utils class to compute the gradients w.r.t. inputs.
        delta_up = delta_up.contiguous()
        # First compute the gradients w.r.t. the pooling operation.
        dldf_pooled = morphpool_cuda.maxpool_backward(delta_up, backward_provenance, out_ks, 1, h, w, device)
        # Then compute it w.r.t. the up-sampling operation.
        dldf = morphpool_cuda.unpool_backward(dldf_pooled, error_provenance, in_ks, stride, device)
        # Also compute the derivative w.r.t. the weights.
        dldh = morphpool_cuda.parameterized_maxpool_backward_h(delta_up, backward_provenance, out_ks, device)
        # Return the gradients w.r.t. the input signal. Return None w.r.t. kernel size, stride and return indices bool.
        return dldf, dldh, None, None, None, None, None, None


class CudaParabolicMorphUnpool2D(CudaMorphUnpool2D):
    """
        Parabolic Morphological unpooling, that is max unpooling, but then also dilate the up-sampled
        signal with a parabolic structuring element.
    """

    def __init__(self, in_channels, input_kernel_size, output_kernel_size, stride=2, init='zero', device: int = 0):
        super(CudaParabolicMorphUnpool2D, self).__init__(input_kernel_size, output_kernel_size, stride=stride)
        self.in_channels = in_channels
        # The parabolic kernels are parameterized by t, with h(z) = -(||z||**2) / 4t where I omit 4, because
        # it is a constant.
        t = torch.empty((in_channels,))
        if init == 'zero':
            # Init to make the centre 0 (it always is, and the corner elements -1).
            torch.nn.init.zeros_(t)
        else:
            torch.nn.init.kaiming_uniform_(t)
        self.t = torch.nn.parameter.Parameter(t, requires_grad=True)
        self.device = device

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        z_i = torch.linspace(-self.out_ks // 2 + 1, self.out_ks // 2, self.out_ks, dtype=torch.float32).to(self.device)
        z_c = z_i.view(-1, 1) ** 2 + z_i.view(1, -1) ** 2
        # Normalize, suhc that the corner elements are 1.
        z_c = z_c / z_c[0, 0]
        # Then repeat for however many kernels we need.
        z = torch.repeat_interleave(z_c.unsqueeze(0), self.in_channels, dim=0)
        # Create the parabolic kernels.
        h = - z * self.t.view(-1, 1, 1)
        # And return the standard CUDA dilation.
        return ParameterizedMorphUnpool2DAutogradFunction.apply(f, h, provenance, size, self.in_ks, self.out_ks,
                                                                self.stride, self.device)
