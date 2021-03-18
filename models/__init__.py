from models.siren import Siren
from models.fourier import FourierNet
from models.wavelet_siren import WaveletSiren

registry = {"siren": Siren, "fourier": FourierNet, "wavelet_siren": WaveletSiren}
