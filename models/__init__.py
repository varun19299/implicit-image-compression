from models.siren import Siren
from models.fourier import FourierNet

registry = {"siren": Siren, "fourier": FourierNet}
