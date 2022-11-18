import torch
import time
import unittest

from morphology.pooling_operations import BaselinePool2D, BaselineParameterizedPool2D, CudaPool2D, CudaParameterizedPool2D


def _sync_and_time():
    torch.cuda.synchronize()
    return time.time()


def _map_cuda_provenances_to_torch_provenances(cuda_provenances, K, b, c, h, w, stride):
    assert K % 2 == 1, "For mapping, kernel size must be odd."
    # Determine point in map.
    h_pad, w_pad = h + K - 1, w + K - 1
    xys = torch.arange(0, h_pad * w_pad, dtype=torch.float32).view(1, 1, h_pad, w_pad).to('cuda:0')
    xys = xys[:, :, (K // 2):-(K // 2):stride, (K // 2):-(K // 2):stride]
    xys = torch.repeat_interleave(torch.repeat_interleave(xys, c, dim=1), b, dim=0)
    xs, ys = xys % w_pad, xys // w_pad
    # Determine provenance location in the kernel.
    prov_ki = cuda_provenances // K - K // 2
    prov_kj = cuda_provenances % K - K // 2
    matched_location = (ys + prov_ki) * w_pad + (xs + prov_kj)
    return matched_location


def _maxpool_test(dims, random=True, verbose=False, epsilon=1e-3):
    device = torch.device('cuda:0')
    b, c, h, w, k, s = dims
    h_out = ((h + (1 if s > 1 else 0)) // s)
    w_out = ((w + (1 if s > 1 else 0)) // s)
    # Get the inputs to the layers.
    if random:
        torch_inputs = torch.rand((b, c, h, w), requires_grad=True, dtype=torch.float32).to(device)
        torch_inputs[0, 0, 1, 0] = 1.2
        torch_inputs[0, 0, 1, 1] = 1.2
        cuda_inputs = torch.zeros_like(torch_inputs, requires_grad=True)
        cuda_inputs.data = torch_inputs.data
        deltas = torch.rand((b, c, h_out, w_out), dtype=torch.float32).to(device)
    else:
        torch_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        cuda_inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
        deltas = torch.arange(1, b * c * h_out * w_out + 1, dtype=torch.float32).reshape(b, c, h_out, w_out).to(device)
    # Declare the two operations we are interested in.
    torch_maxpool = BaselinePool2D(k, stride=s)
    cuda_maxpool = CudaPool2D(k, stride=s).to(device)
    # Perform forward operations.
    t1 = _sync_and_time()
    torch_out, torch_provenances = torch_maxpool(torch_inputs)
    t2 = _sync_and_time()
    cuda_out, cuda_provenances = cuda_maxpool(cuda_inputs)
    t3 = _sync_and_time()
    forward_success = (torch.abs(torch_out - cuda_out) < epsilon).all()
    # Now perform the backward operations.
    torch_loss = (torch_out * deltas).sum()
    t4 = _sync_and_time()
    torch_delta_down = torch.autograd.grad(torch_loss, torch_inputs, retain_graph=True)[0]
    t5 = _sync_and_time()
    cuda_loss = (cuda_out * deltas).sum()
    t6 = _sync_and_time()
    cuda_delta_down = torch.autograd.grad(cuda_loss, cuda_inputs, retain_graph=True)[0]
    t7 = _sync_and_time()
    backward_success = (torch.abs(torch_delta_down - cuda_delta_down) < epsilon).all()
    if verbose:
        print(f"MAX POOL FORWARD SUCCESS: {forward_success}")
        print(f"MAX POOL BACKWARD SUCCESS: {backward_success}")
        print(f'TORCH: {t5 - t4:.6f} | CUDA {t7 - t6:.6f}')
        print(f'TORCH: {t2 - t1:.6f} | CUDA {t3 - t2:.6f}')
    return forward_success and backward_success


def _baseline_versus_parameterized_torch_test(dims, random=True):
    device = torch.device('cuda:0')
    b, c, h, w, k, s = dims
    if random:
        inputs = torch.rand((b, c, h, w), requires_grad=True, dtype=torch.float32).to(device)
    else:
        inputs = torch.arange(0, b * c * h * w, requires_grad=True, dtype=torch.float32).reshape((b, c, h, w)).to(device)
    # Perform a parameterized pool, but with zero-initialized weights.
    torch_pool = BaselineParameterizedPool2D(c, k, s, init='zero').to(device)
    torch_outputs, _ = torch_pool(inputs)
    # Perform the unparameterized baseline, which should equal the zero weights.
    torch_nonparam_pool = BaselinePool2D(k, s).to(device)
    torch_nonparam_outputs, _ = torch_nonparam_pool(inputs)
    if torch_outputs.shape != torch_nonparam_outputs.shape:
        return False
    return (torch.abs(torch_outputs - torch_nonparam_outputs) < 1e-4).all()


def _parameterized_maxpool_test(dims, random=True, verbose=False, epsilon=1e-3):
    device = torch.device('cuda:0')
    b, c, h, w, k, s = dims
    h_out = ((h + (1 if s > 1 else 0)) // s)
    w_out = ((w + (1 if s > 1 else 0)) // s)
    # These are the torch and cuda version of parameterized max pool.
    torch_pool = BaselineParameterizedPool2D(c, k, s).to(device)
    cuda_pool = CudaParameterizedPool2D(c, k, s).to(device)
    # Initialize the inputs to the forward and backward modules.
    if random:
        torch_inputs = torch.rand((b, c, h, w), dtype=torch.float32, requires_grad=True).to(device)
        se = torch.rand((c, k, k), dtype=torch.float32).to(device)
        deltas = torch.rand((b, c, h_out, w_out), dtype=torch.float32).to(device)
    else:
        torch_inputs = torch.arange(0, b * c * h * w, dtype=torch.float32,
                                    requires_grad=True, device=device).view((b, c, h, w))
        se = torch.arange(0, c * k ** 2, dtype=torch.float32).reshape(c, k, k).to(device)
        deltas = torch.arange(1, b * c * h_out * w_out + 1, dtype=torch.float32).reshape(b, c, h_out, w_out).to(device)
    # Set the inputs and structuring elements to have the same value.
    cuda_inputs = torch.zeros_like(torch_inputs, requires_grad=True)
    cuda_inputs.data = torch_inputs.data
    cuda_pool.h.data = se.data
    torch_pool.h.data = torch.movedim(se.data, 0, 2).view(1, k ** 2, c)
    # Forward operation.
    t1 = _sync_and_time()
    cuda_outputs, cuda_provenances = cuda_pool(cuda_inputs)
    t2 = _sync_and_time()
    torch_outputs, torch_provenances = torch_pool(torch_inputs)
    t3 = _sync_and_time()
    forward_output_success = (torch.abs(torch_outputs - cuda_outputs) < 1e-3).all()
    forward_provenance_success = (torch_provenances == cuda_provenances).all()
    # Compute the loss and back-propagate it for torch.
    torch_loss = (torch_outputs * deltas).sum()
    torch_dldf = torch.autograd.grad(torch_loss, torch_inputs, retain_graph=True)[0]
    t4 = _sync_and_time()
    torch_loss.backward(retain_graph=False)
    t5 = _sync_and_time()
    # For cuda.
    cuda_loss = (cuda_outputs * deltas).sum()
    cuda_dldf = torch.autograd.grad(cuda_loss, cuda_inputs, retain_graph=True)[0]
    t6 = _sync_and_time()
    cuda_loss.backward(retain_graph=False)
    t7 = _sync_and_time()
    # Return whether the operation was a success.
    dldf_success = (torch.abs(torch_dldf - cuda_dldf) < epsilon).all()
    dldh_success = (torch.abs(torch.movedim(torch_pool.h.grad.data, 2, 0).view(c, k, k) - cuda_pool.h.grad.data) < epsilon).all()
    if verbose:
        print(f"FORWARD\n"
              f"outs: {forward_output_success} | prov: {forward_provenance_success}\n"
              f"TORCH: {t2 - t1:.6f} | CUDA {t3 - t2:.6f}")
        print(f"BACKWARD\n"
              f"dldf: {dldf_success} | dldh: {dldh_success}\n"
              f"TORCH: {t5 - t4:.6f} | CUDA {t7 - t6:.6f}")
    return forward_output_success and forward_provenance_success and dldh_success and dldh_success


class PoolTests(unittest.TestCase):

    def test_unparameterized_pool_stride2_evenkernel(self):
        """ Tests whether unparameterized pooling with even kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 2), (4, 8, 64, 63, 2, 2), (4, 8, 64, 64, 2, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_pool_stride2_oddkernel(self):
        """ Tests whether unparameterized pooling with odd kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 2), (4, 8, 64, 63, 3, 2), (4, 8, 64, 64, 3, 2),
            (4, 8, 63, 64, 5, 2), (4, 8, 64, 63, 5, 2), (4, 8, 64, 64, 5, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_pool_stride1_evenkernel(self):
        """ Tests whether unparameterized pooling with even kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 1), (4, 8, 64, 63, 2, 1), (4, 8, 64, 64, 2, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_unparameterized_pool_stride1_oddkernel(self):
        """ Tests whether unparameterized pooling with odd kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 1), (4, 8, 64, 63, 3, 1), (4, 8, 64, 64, 3, 1),
            (4, 8, 63, 64, 5, 1), (4, 8, 64, 63, 5, 1), (4, 8, 64, 64, 5, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_versus_parameterized_pool_torch(self):
        """ Tests whether zero-initialization of parameterized pool in torch code yields the same
            results as unparameterized pool in torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 1), (4, 8, 64, 64, 3, 1), (4, 8, 63, 64, 3, 2), (4, 8, 64, 64, 3, 2),
            (4, 8, 63, 64, 2, 2), (4, 8, 64, 64, 2, 1), (4, 8, 63, 64, 2, 2), (4, 8, 64, 64, 2, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_baseline_versus_parameterized_torch_test(dimensions_set, random=True))

    def test_parameterized_pool_stride2_evenkernel(self):
        """ Tests whether parameterized pooling with even kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 2), (4, 8, 64, 63, 2, 2), (4, 8, 64, 64, 2, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_pool_stride2_oddkernel(self):
        """ Tests whether parameterized pooling with odd kernel size and a stride of 2 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 2), (4, 8, 64, 63, 3, 2), (4, 8, 64, 64, 3, 2),
            (4, 8, 63, 64, 5, 2), (4, 8, 64, 63, 5, 2), (4, 8, 64, 64, 5, 2),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_pool_stride1_evenkernel(self):
        """ Tests whether parameterized pooling with even kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 2, 1), (4, 8, 64, 63, 2, 1), (4, 8, 64, 64, 2, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_maxpool_test(dimensions_set, random=True, verbose=False))

    def test_parameterized_pool_stride1_oddkernel(self):
        """ Tests whether parameterized pooling with odd kernel size and a stride of 1 yields
            the same result in CUDA and torch code.
        """
        dimensions_for_testing = [
            (4, 8, 63, 64, 3, 1), (4, 8, 64, 63, 3, 1), (4, 8, 64, 64, 3, 1),
            (4, 8, 63, 64, 5, 1), (4, 8, 64, 63, 5, 1), (4, 8, 64, 64, 5, 1),
        ]
        for dimensions_set in dimensions_for_testing:
            self.assertTrue(_parameterized_maxpool_test(dimensions_set, random=True, verbose=False))


if __name__ == '__main__':
    unittest.main()


