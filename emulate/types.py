from enum import IntEnum, unique as uniqueEnum, Enum


class BoundaryCondition(Enum):
    INCOMING = -1  # -ie
    OUTGOING = +1  # +ie
    STANDING = +0  # principal value


@uniqueEnum
class QuadratureType(IntEnum):
    GaussLegendre = 0
    GaussLobatto = 1
    SemiInfinite = 2
    ExponentialGaussLegendre = 3
    ExponentialGaussLobatto = 4

    @classmethod
    def from_suffix(cls, suffix):
        suffix = str(suffix).lower()
        if suffix == "l":
            return cls.GaussLobatto
        if suffix == "i":
            return cls.SemiInfinite
        if suffix == "e":
            return cls.ExponentialGaussLegendre
        if suffix == "f":
            return cls.ExponentialGaussLobatto
        return cls.GaussLegendre
