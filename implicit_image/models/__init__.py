from implicit_image.models.siren import Siren
from implicit_image.models.fourier import FourierNet
from implicit_image.models.wavelet_siren import WaveletSiren

registry = {"siren": Siren, "fourier": FourierNet, "wavelet_siren": WaveletSiren}
