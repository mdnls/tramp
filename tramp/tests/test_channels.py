import unittest
from tramp.channels import (
    AbsChannel, SgnChannel, ReluChannel, LeakyReluChannel, HardTanhChannel,
    MultiConvChannel, LinearChannel, DiagonalChannel, UpsampleChannel
)
from tramp.ensembles import Multi2dConvEnsemble
import numpy as np
import torch


def empirical_second_moment(tau_z, channel):
    """
    Estimate second_moment by sampling.
    """
    noise = np.random.standard_normal(size=1000 * 1000)
    Z = np.sqrt(tau_z) * noise / noise.std()
    X = channel.sample(Z)
    tau_x = (X**2).mean()
    return tau_x


def explicit_integral(az, bz, ax, bx, channel):
    """
    Compute rx, vx, rz, vz for p(x|z) by integration
    """
    def belief(z, x):
        L = -0.5 * ax * (x**2) + bx * x - 0.5 * az * (z**2) + bz * z
        return np.exp(L)

    def z_belief(z, x):
        return z * belief(z, x)

    def z2_belief(z, x):
        return (z**2) * belief(z, x)

    def x_belief(z, x):
        return x * belief(z, x)

    def x2_belief(z, x):
        return (x**2) * belief(z, x)

    zmin = bz / az - 10 / np.sqrt(az)
    zmax = bz / az + 10 / np.sqrt(az)

    Z = channel.measure(belief, zmin, zmax)
    rx = channel.measure(x_belief, zmin, zmax) / Z
    x2 = channel.measure(x2_belief, zmin, zmax) / Z
    vx = x2 - rx**2
    rz = channel.measure(z_belief, zmin, zmax) / Z
    z2 = channel.measure(z2_belief, zmin, zmax) / Z
    vz = z2 - rz**2

    return rz, vz, rx, vx


class ChannelsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            dict(az=2.1, bz=2.0, ax=2.0, bx=2.0, tau_z=2.0),
            dict(az=2.0, bz=+1.6, ax=1.5, bx=1.3, tau_z=1.5),
            dict(az=2.0, bz=-1.6, ax=1.5, bx=1.3, tau_z=1.0)
        ]

    def tearDown(self):
        pass

    def _test_function_second_moment(self, channel, records, places=6):
        for record in records:
            tau_z = record["tau_z"]
            tau_x_emp = empirical_second_moment(tau_z, channel)
            tau_x_hat = channel.second_moment(tau_z)
            msg = f"record={record}"
            self.assertAlmostEqual(tau_x_emp, tau_x_hat, places=places, msg=msg)

    def _test_function_posterior(self, channel, records, places=12):
        for record in records:
            az, bz, ax, bx = record["az"], record["bz"], record["ax"], record["bx"]
            rz, vz, rx, vx = explicit_integral(az, bz, ax, bx, channel)
            rx_hat, vx_hat = channel.compute_forward_posterior(az, bz, ax, bx)
            rz_hat, vz_hat = channel.compute_backward_posterior(az, bz, ax, bx)
            msg = f"record={record}"
            self.assertAlmostEqual(rx, rx_hat, places=places, msg=msg)
            self.assertAlmostEqual(vx, vx_hat, places=places, msg=msg)
            self.assertAlmostEqual(rz, rz_hat, places=places, msg=msg)
            self.assertAlmostEqual(vz, vz_hat, places=places, msg=msg)

    def _test_function_proba(self, channel, records, places=12):
        for record in records:
            az, ax, tau_z = record["az"], record["ax"], record["tau_z"]
            def one(bz, bx): return 1
            sum_proba = channel.beliefs_measure(az, ax, tau_z, f=one)
            msg = f"record={record}"
            self.assertAlmostEqual(sum_proba, 1., places=places, msg=msg)

    def test_abs_posterior(self):
        channel = AbsChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_sgn_posterior(self):
        channel = SgnChannel()
        self._test_function_posterior(channel, self.records, places=4)

    def test_relu_posterior(self):
        channel = ReluChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_leaky_relu_posterior(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_posterior(channel, self.records, places=6)

    def test_hard_tanh_posterior(self):
        channel = HardTanhChannel()
        self._test_function_posterior(channel, self.records, places=1)

    def test_abs_second_moment(self):
        channel = AbsChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_sgn_second_moment(self):
        channel = SgnChannel()
        self._test_function_second_moment(channel, self.records)

    def test_relu_second_moment(self):
        channel = ReluChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_leaky_relu_second_moment(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_second_moment(channel, self.records, places=2)

    def test_hard_tanh_second_moment(self):
        channel = HardTanhChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_abs_proba(self):
        channel = AbsChannel()
        self._test_function_proba(channel, self.records)

    def test_sgn_proba(self):
        channel = SgnChannel()
        self._test_function_proba(channel, self.records)

    def test_relu_proba(self):
        channel = ReluChannel()
        self._test_function_proba(channel, self.records)

    def test_leaky_relu_proba(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_proba(channel, self.records)


class MultiConvChannelTest(unittest.TestCase):
    def setUp(self):
        H, W = (10, 11) # height, width
        k= 3
        M, N = (3, 4) # out channels, in channels

        self.inp_imdim = (N, H, W)
        self.outp_imdim = (M, H, W)

        # Generate the convolution
        self.inp_img = np.random.normal(size=(N, H, W))
        self.conv_ensemble = Multi2dConvEnsemble(width=W, height=H, in_channels=N, out_channels=M, k=3)
        conv_filter = self.conv_ensemble.generate(with_filter=True)

        self.conv_filter = conv_filter
        self.mcc_channel = MultiConvChannel(self.conv_filter, block_shape=(H, W))
        self.dense_conv = self.mcc_channel.densify()

        # Construct reference implementations of convolutions and linear operators
        self.ref_conv = torch.nn.Conv2d(in_channels=N, out_channels=M, padding_mode="circular", padding=(k - 1) // 2,
                                    kernel_size=k, bias=False, stride=1)
        self.ref_conv.weight.data = torch.FloatTensor(conv_filter)
        inp_img_pt = torch.FloatTensor(self.inp_img[np.newaxis, ...])
        ref_outp_img = self.ref_conv(inp_img_pt).detach().cpu().numpy()[0]
        self.ref_outp_img = ref_outp_img

        self.ref_linear = LinearChannel(W=self.dense_conv)

    def tearDown(self):
        pass

    def test_unitary(self):
        # Test the closed form SVD and the virtual matrix multiplication by verifying unitarity
        Vt_img = self.mcc_channel.V.T(self.inp_img)
        Ut_img = self.mcc_channel.U.T(self.ref_outp_img)

        self.assertTrue(np.isclose(np.linalg.norm(self.inp_img), np.linalg.norm(Vt_img)))
        self.assertTrue(np.isclose(np.linalg.norm(self.ref_outp_img), np.linalg.norm(Ut_img)))
        self.assertTrue(np.allclose(self.inp_img, self.mcc_channel.V(Vt_img)))
        self.assertTrue(np.allclose(self.ref_outp_img, self.mcc_channel.U(Ut_img)))

    def test_densify(self):
        # Test that densify() returns dense matrices that correctly implement sparse matrix behavior
        Vt_img = self.mcc_channel.V.T(self.inp_img)
        Ut_img = self.mcc_channel.U.T(self.ref_outp_img)
        self.assertTrue(np.allclose( self.mcc_channel.V.densify() @ Vt_img.flatten(), self.inp_img.flatten()))
        self.assertTrue(np.allclose( self.mcc_channel.U.densify() @ Ut_img.flatten(), self.ref_outp_img.flatten()))
        self.assertTrue(np.allclose( self.mcc_channel.densify() @ self.inp_img.flatten(), self.mcc_channel.at(self.inp_img).flatten()))

    def test_conv_agreement(self):
        # Test the sparse matrix mult matches a pytorch 2d convolution
        C = self.mcc_channel
        outp_img = C.at(self.inp_img)
        self.assertTrue(np.allclose(outp_img, self.ref_outp_img, atol=1e-6))

    def test_linear_agreement(self):
        # Check that the multichannel conv channel exactly matches the behavior of the corresponding
        #   dense linear channel.
        az = np.random.uniform(low=1, high=5)
        ax = np.random.uniform(low=1, high=5)
        tau_z = np.random.uniform(low=1, high=5)
        bz = np.random.normal(size=self.inp_imdim)
        bx = np.random.normal(size=self.outp_imdim)

        self.assertTrue(np.allclose(self.mcc_channel.sample(bz).flatten(),
                                    self.ref_linear.sample(bz.flatten())))

        self.assertTrue(np.allclose(self.mcc_channel.compute_forward_variance(az, ax),
                                    self.ref_linear.compute_forward_variance(az, ax)))

        self.assertTrue(np.allclose(self.mcc_channel.compute_backward_variance(az, ax),
                                    self.ref_linear.compute_backward_variance(az, ax)))

        self.assertTrue(np.allclose(self.mcc_channel.compute_backward_mean(az, bz, ax, bx).flatten(),
                                    self.ref_linear.compute_backward_mean(az, bz.flatten(), ax, bx.flatten())))

        self.assertTrue(np.allclose(self.mcc_channel.compute_log_partition(az, bz, ax, bx),
                                    self.ref_linear.compute_log_partition(az, bz.flatten(), ax, bx.flatten())))

        self.assertTrue(np.allclose(self.mcc_channel.compute_mutual_information(az, ax, tau_z),
                                    self.ref_linear.compute_mutual_information(az, ax, tau_z)))

class DiagonalChannelTest(unittest.TestCase):
    def setUp(self):
        self.dim = (32, 32)
        self.S = np.random.normal(size=self.dim)
        self.channel = DiagonalChannel(S=self.S)
        self.ref_linear = LinearChannel(W=np.diag(self.S.flatten()))

    def test_linear_agreement(self):
        az = np.random.uniform(low=1, high=5)
        ax = np.random.uniform(low=1, high=5)
        tau_z = np.random.uniform(low=1, high=5)
        bz = np.random.normal(size=self.dim)
        bx = np.random.normal(size=self.dim)

        self.assertTrue(np.allclose(self.channel.sample(bz).flatten(),
                                    self.ref_linear.sample(bz.flatten())))

        self.assertTrue(np.allclose(self.channel.compute_forward_variance(az, ax),
                                    self.ref_linear.compute_forward_variance(az, ax)))

        self.assertTrue(np.allclose(self.channel.compute_backward_variance(az, ax),
                                    self.ref_linear.compute_backward_variance(az, ax)))

        self.assertTrue(np.allclose(self.channel.compute_backward_mean(az, bz, ax, bx).flatten(),
                                    self.ref_linear.compute_backward_mean(az, bz.flatten(), ax, bx.flatten())))

        self.assertTrue(np.allclose(self.channel.compute_log_partition(az, bz, ax, bx),
                                    self.ref_linear.compute_log_partition(az, bz.flatten(), ax, bx.flatten())))

        self.assertTrue(np.allclose(self.channel.compute_mutual_information(az, ax, tau_z),
                                    self.ref_linear.compute_mutual_information(az, ax, tau_z)))


class UpsampleChannelTest(unittest.TestCase):
    def setUp(self):
        self.example_image = np.random.normal(size=(3, 32, 32))
        self.channel = UpsampleChannel(input_shape=(3, 32, 32), output_shape=(3, 64, 64))
        self.ref_upsample_operator = torch.nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)

    def test_upsample_agreement(self):
        ref_ups_image = self.ref_upsample_operator(torch.FloatTensor(self.example_image[np.newaxis, ...]))[0].detach().numpy()
        ups_image = self.channel.sample(self.example_image)
        self.assertTrue(np.allclose(ups_image, ref_ups_image, atol=1e-6))

if __name__ == "__main__":

    unittest.main()
