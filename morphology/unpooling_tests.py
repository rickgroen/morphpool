import torch
import time
import unittest

from morphology.pooling_operations import BaselinePool2D, CudaPool2D
from morphology.unpooling_operations import BaselineMaxUnpool2D, BaselineUnpool2D, \
    BaselineParameterizedUnpool2D, CudaMaxUnpool2D, CudaMorphUnpool2D, CudaParameterizedMorphUnpool2D


def _sync_and_time():
    torch.cuda.synchronize()
    return time.time()


def _unpool_test(dims, random=True, verbose=False, epsilon=1e-4):
    device = torch.device('cuda:0')
    b, c, h, w, k_in, s = dims
    # Get the inputs to the layers.
    if random:
        torch_inputs = torch.rand((b, c, h, w), requires_grad=True, dtype=torch.float32).to(device)
        cuda_inputs = torch.zeros_like(torch_inputs, requires_grad=True)
        cuda_inputs.data = torch_inputs.data
        deltas = torch.rand((b, c, h, w), dtype=torch.float32).to(device)
    else:
        torch_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(
            device)
        cuda_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(
            device)
        deltas = torch.arange(1, b * c * h * w + 1, dtype=torch.float32).reshape(b, c, h, w).to(device)
    # As a first step, we need to do downsampling from a maxpool to get the correct indices.
    torch_pool = BaselinePool2D(k_in, stride=s).to(device)
    torch_downsampled, torch_indices_downsampled = torch_pool(torch_inputs)
    cuda_pool = CudaPool2D(k_in, stride=s).to(device)
    cuda_downsampled, cuda_indices_downsampled = cuda_pool(cuda_inputs)
    # Declare the two operations we are interested in.
    torch_unpool = BaselineMaxUnpool2D(k_in, stride=s).to(device)
    cuda_unpool = CudaMaxUnpool2D(k_in, stride=s).to(device)
    # Perform forward operations.
    t1 = _sync_and_time()
    torch_out = torch_unpool(torch_downsampled, torch_indices_downsampled, size=(h, w))
    t2 = _sync_and_time()
    cuda_out = cuda_unpool(cuda_downsampled, cuda_indices_downsampled, (h, w))
    t3 = _sync_and_time()
    forward_success = (torch.abs(torch_out - cuda_out) < epsilon).all()
    # Now perform the backward operations.
    torch_loss = (torch_out * deltas).sum()
    t4 = _sync_and_time()
    torch_delta_down = torch.autograd.grad(torch_loss, torch_downsampled, retain_graph=True)[0]
    t5 = _sync_and_time()
    cuda_loss = (cuda_out * deltas).sum()
    t6 = _sync_and_time()
    cuda_delta_down = torch.autograd.grad(cuda_loss, cuda_downsampled, retain_graph=True)[0]
    t7 = _sync_and_time()
    backward_success = (torch.abs(torch_delta_down - cuda_delta_down) < epsilon).all()
    if verbose:
        print(f"MAX POOL FORWARD SUCCESS: {forward_success}")
        print(f"MAX POOL BACKWARD SUCCESS: {backward_success}")
        print(f'TORCH: {t5 - t4:.6f} | CUDA {t7 - t6:.6f}')
        print(f'TORCH: {t2 - t1:.6f} | CUDA {t3 - t2:.6f}')
    return forward_success and backward_success


def _morph_unpool_test(dims, random=True, verbose=False, epsilon=1e-4):
    """
        This actually does work, but raised an interesting question:
        In your maxpool backward function do you
        - take the mean of incoming errors (this seems to be torch's behaviour).
        - take the sum of the incoming errors (this is what I currently do in CUDA)
        - take the max of the incoming errors (this would be most in line with our previous paper.)
    """
    device = torch.device('cuda:0')
    b, c, h, w, k_in, k_out, s = dims
    # Get the inputs to the layers.
    if random:
        torch_inputs = torch.rand((b, c, h, w), requires_grad=True, dtype=torch.float32).to(device)
        cuda_inputs = torch.zeros_like(torch_inputs, requires_grad=True)
        cuda_inputs.data = torch_inputs.data
        deltas = torch.rand((b, c, h, w), dtype=torch.float32).to(device)
    else:
        torch_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        cuda_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        deltas = torch.arange(1, b * c * h * w + 1, dtype=torch.float32).reshape(b, c, h, w).to(device)
    # As a first step, we need to do downsampling from a maxpool to get the correct indices.
    torch_pool = BaselinePool2D(k_in, stride=s).to(device)
    torch_downsampled, torch_indices_downsampled = torch_pool(torch_inputs)
    cuda_pool = CudaPool2D(k_in, stride=s).to(device)
    cuda_downsampled, cuda_indices_downsampled = cuda_pool(cuda_inputs)
    # Declare the two operations we are interested in.
    torch_unpool = BaselineUnpool2D(k_in, k_out, stride=s).to(device)
    cuda_unpool = CudaMorphUnpool2D(k_in, k_out, stride=s).to(device)
    # Perform forward operations.
    t1 = _sync_and_time()
    torch_out = torch_unpool(torch_downsampled, torch_indices_downsampled, size=(h, w))
    t2 = _sync_and_time()
    cuda_out = cuda_unpool(cuda_downsampled, cuda_indices_downsampled, (h, w))
    t3 = _sync_and_time()
    forward_success = (torch.abs(torch_out - cuda_out) < epsilon).all()
    # Now perform the backward operations.
    torch_loss = (torch_out * deltas).sum()
    t4 = _sync_and_time()
    torch_delta_down = torch.autograd.grad(torch_loss, torch_downsampled, retain_graph=True)[0]
    t5 = _sync_and_time()
    cuda_loss = (cuda_out * deltas).sum()
    t6 = _sync_and_time()
    cuda_delta_down = torch.autograd.grad(cuda_loss, cuda_downsampled, retain_graph=True)[0]
    t7 = _sync_and_time()
    backward_success = (torch.abs(torch_delta_down - cuda_delta_down) < epsilon).all()
    if verbose:
        print(f"MAX POOL FORWARD SUCCESS: {forward_success}")
        print(f"MAX POOL BACKWARD SUCCESS: {backward_success}")
        print(f'TORCH: {t5 - t4:.6f} | CUDA {t7 - t6:.6f}')
        print(f'TORCH: {t2 - t1:.6f} | CUDA {t3 - t2:.6f}')
    return forward_success and backward_success


def _parameterized_morph_unpool_test(dims, random=True, verbose=False, epsilon=1e-4):
    device = torch.device('cuda:0')
    b, c, h, w, k_in, k_out, s = dims
    # Get the inputs to the layers.
    if random:
        torch_inputs = torch.rand((b, c, h, w), requires_grad=True, dtype=torch.float32).to(device)
        cuda_inputs = torch.zeros_like(torch_inputs, requires_grad=True)
        cuda_inputs.data = torch_inputs.data
        se = torch.rand((c, k_out, k_out), dtype=torch.float32).to(device)
        deltas = torch.rand((b, c, h, w), dtype=torch.float32).to(device)
    else:
        torch_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        cuda_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        se = torch.arange(0, c * k_out ** 2, dtype=torch.float32).reshape(c, k_out, k_out).to(device)
        deltas = torch.arange(1, b * c * h * w + 1, dtype=torch.float32).reshape(b, c, h, w).to(device)
    # As a first step, we need to do downsampling from a maxpool to get the correct indices.
    torch_pool = BaselinePool2D(k_in, stride=s).to(device)
    torch_downsampled, torch_indices_downsampled = torch_pool(torch_inputs)
    cuda_pool = CudaPool2D(k_in, stride=s).to(device)
    cuda_downsampled, cuda_indices_downsampled = cuda_pool(cuda_inputs)
    # Declare the two operations we are interested in.
    torch_unpool = BaselineParameterizedUnpool2D(c, k_in, k_out, stride=s).to(device)
    torch_unpool.h.data = se.view(1, c, k_out ** 2, 1)
    cuda_unpool = CudaParameterizedMorphUnpool2D(c, k_in, k_out, stride=s).to(device)
    cuda_unpool.h.data = se
    # Perform forward operations.
    t1 = _sync_and_time()
    torch_out, intermediate = torch_unpool(torch_downsampled, torch_indices_downsampled, size=(h, w))
    t2 = _sync_and_time()
    cuda_out = cuda_unpool(cuda_downsampled, cuda_indices_downsampled, (h, w))
    t3 = _sync_and_time()
    forward_success = (torch.abs(torch_out - cuda_out) < epsilon).all()
    # Now perform the backward operations.
    torch_loss = (torch_out * deltas).sum()
    t4 = _sync_and_time()
    torch_dldf = torch.autograd.grad(torch_loss, torch_downsampled, retain_graph=True)[0]
    t5 = _sync_and_time()
    cuda_loss = (cuda_out * deltas).sum()
    t6 = _sync_and_time()
    cuda_dldf = torch.autograd.grad(cuda_loss, cuda_downsampled, retain_graph=True)[0]
    t7 = _sync_and_time()
    dldf_success = (torch.abs(torch_dldf - cuda_dldf) < epsilon).all()
    torch_loss.backward(retain_graph=False)
    cuda_loss.backward(retain_graph=False)
    # Define the success here as 0.01, because we get some precision loss in either module.
    dldh_success = (torch.abs(torch_unpool.h.grad.data.view(c, k_out, k_out) - cuda_unpool.h.grad.data) < 0.01).all()
    if verbose:
        print(f"FORWARD\n"
              f"outs: {forward_success}\n"
              f"TORCH: {t2 - t1:.6f} | CUDA {t3 - t2:.6f}")
        print(f"BACKWARD\n"
              f"dldf: {dldf_success} | dldh: {dldh_success}\n"
              f"TORCH: {t5 - t4:.6f} | CUDA {t7 - t6:.6f}")
    return forward_success and dldf_success and dldh_success


class UnpoolTests(unittest.TestCase):

    def test_max_unpool_stride2_evenkernel(self):
        """ Tests whether the nn.Unpool2D torch' equivalent unpooling with even kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 2), (4, 8, 64, 63, 2, 2), (4, 8, 64, 64, 2, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_unpool_test(dimensions_set, random=True, verbose=False))

    def test_max_unpool_stride2_oddkernel(self):
        """ Tests whether the nn.Unpool2D torch' equivalent unpooling with odd kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 2), (4, 8, 64, 63, 3, 2), (4, 8, 64, 64, 3, 2),
            (4, 8, 63, 64, 5, 2), (4, 8, 64, 63, 5, 2), (4, 8, 64, 64, 5, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_unpool_test(dimensions_set, random=True, verbose=False))

    def test_max_unpool_stride1_evenkernel(self):
        """ Tests whether the nn.Unpool2D torch' equivalent unpooling with even kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 1), (4, 8, 64, 63, 2, 1), (4, 8, 64, 64, 2, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_unpool_test(dimensions_set, random=True, verbose=False))

    def test_max_unpool_stride1_oddkernel(self):
        """ Tests whether the nn.Unpool2D torch' equivalent unpooling with odd kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 1), (4, 8, 64, 63, 3, 1), (4, 8, 64, 64, 3, 1),
            (4, 8, 63, 64, 5, 1), (4, 8, 64, 63, 5, 1), (4, 8, 64, 64, 5, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_unpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_morph_unpool_stride2_evenkernel(self):
        """ Tests whether unparameterized morphological unpooling with even kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 3, 2), (4, 8, 64, 63, 2, 3, 2), (4, 8, 64, 64, 2, 3, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_morph_unpool_stride2_oddkernel(self):
        """ Tests whether unparameterized morphological unpooling with odd kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 3, 2), (4, 8, 64, 63, 3, 3, 2), (4, 8, 64, 64, 3, 3, 2),
            (4, 8, 63, 64, 5, 3, 2), (4, 8, 64, 63, 5, 3, 2), (4, 8, 64, 64, 5, 3, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_morph_unpool_stride1_evenkernel(self):
        """ Tests whether unparameterized morphological unpooling with even kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 3, 1), (4, 8, 64, 63, 2, 3, 1), (4, 8, 64, 64, 2, 3, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_morph_unpool_stride1_oddkernel(self):
        """ Tests whether unparameterized morphological unpooling with odd kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 3, 1), (4, 8, 64, 63, 3, 3, 1), (4, 8, 64, 64, 3, 3, 1),
            (4, 8, 63, 64, 5, 3, 1), (4, 8, 64, 63, 5, 3, 1), (4, 8, 64, 64, 5, 3, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_morph_unpool_stride2_evenkernel(self):
        """ Tests whether parameterized morphological unpooling with even kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 3, 2), (4, 8, 64, 63, 2, 3, 2), (4, 8, 64, 64, 2, 3, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_morph_unpool_stride2_oddkernel(self):
        """ Tests whether parameterized morphological unpooling with odd kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 3, 2), (4, 8, 64, 63, 3, 3, 2), (4, 8, 64, 64, 3, 3, 2),
            (4, 8, 63, 64, 5, 3, 2), (4, 8, 64, 63, 5, 3, 2), (4, 8, 64, 64, 5, 3, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_morph_unpool_stride1_evenkernel(self):
        """ Tests whether parameterized morphological unpooling with even kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 3, 1), (4, 8, 64, 63, 2, 3, 1), (4, 8, 64, 64, 2, 3, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_morph_unpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_morph_unpool_stride1_oddkernel(self):
        """ Tests whether parameterized morphological unpooling with odd kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 3, 1), (4, 8, 64, 63, 3, 3, 1), (4, 8, 64, 64, 3, 3, 1),
            (4, 8, 63, 64, 5, 3, 1), (4, 8, 64, 63, 5, 3, 1), (4, 8, 64, 64, 5, 3, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_morph_unpool_test(dimensions_set, random=True, verbose=False))


if __name__ == '__main__':
    unittest.main()


