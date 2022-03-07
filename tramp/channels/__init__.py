# noise
from .noise.gaussian_channel import GaussianChannel
# shape
from .shape.concat_channel import ConcatChannel
from .shape.duplicate_channel import DuplicateChannel
from .shape.reshape_channel import ReshapeChannel
# linear
from .linear.sum_channel import SumChannel
from .linear.dft_channel import DFTChannel
from .linear.bias_channel import BiasChannel
from .linear.rotation_channel import RotationChannel
from .linear.unitary_channel import UnitaryChannel
from .linear.linear_channel import LinearChannel, DiagonalChannel, ColorwiseLinearChannel
from .linear.aniso_linear_channel import AnisoLinearChannel
from .linear.complex_linear_channel import ComplexLinearChannel
from .linear.product_channel import ProductChannel, MC_ProductChannel
from .linear.aniso_to_iso import AnisoToIso, IsoToAniso
from .linear.conv_channel import (
    ConvChannel, BatchConvChannel, Blur1DChannel, Blur2DChannel,
    DifferentialChannel, LaplacianChannel, MultiConvChannel
)
from .linear.gradient_channel import GradientChannel
from .linear.analytical_linear_channel import (
    AnalyticalLinearChannel, MarchenkoPasturChannel
)
# activation
from .activation.piecewise_linear_channel import (
    PiecewiseLinearChannel, SgnChannel, AbsChannel, AsymmetricAbsChannel,
    ReluChannel, LeakyReluChannel, HardTanhChannel, HardSigmoidChannel,
    SymmetricDoorChannel
)
from .activation.tanh_channel import TanhChannel
from .activation.modulus_channel import ModulusChannel
# low rank
from .low_rank.low_rank_gram_channel import LowRankGramChannel
from .low_rank.low_rank_factorization import LowRankFactorization
from .linear.upsampling_channel import UpsampleChannel

CHANNEL_CLASSES = {
    "gaussian": GaussianChannel,
    "concat": ConcatChannel,
    "duplicate": DuplicateChannel,
    "reshape": ReshapeChannel,
    "sum": SumChannel,
    "dft": DFTChannel,
    "bias": BiasChannel,
    "rotation": RotationChannel,
    "unitary": UnitaryChannel,
    "linear": LinearChannel,
    "complex_linear": ComplexLinearChannel,
    "conv": ConvChannel,
    "batchconv": BatchConvChannel,
    "blur_1d": Blur1DChannel,
    "blur_2d": Blur2DChannel,
    "diff": DifferentialChannel,
    "laplacian": LaplacianChannel,
    "gradient": GradientChannel,
    "analytical": AnalyticalLinearChannel,
    "marchenko": MarchenkoPasturChannel,
    "sgn": SgnChannel,
    "abs": AbsChannel,
    "a-abs": AsymmetricAbsChannel,
    "relu": ReluChannel,
    "l-relu": LeakyReluChannel,
    "h-tanh": HardTanhChannel,
    "h-sigm": HardSigmoidChannel,
    "door": SymmetricDoorChannel,
    "modulus": ModulusChannel,
    "multiconv": MultiConvChannel,
    "diagonal": DiagonalChannel,
    "colorwise": ColorwiseLinearChannel,
    "upsample": UpsampleChannel,
    "product": ProductChannel,
    "mc_product": MC_ProductChannel,
    "aniso2iso": AnisoToIso,
    "iso2aniso": IsoToAniso,
    "aniso_linear": AnisoLinearChannel
}


def get_channel(channel_type, **kwargs):
    channel = CHANNEL_CLASSES[channel_type](**kwargs)
    return channel
