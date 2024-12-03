"""Helpfunctions for calculations with electromagnetic signals."""

import enum

import numpy as np


class PowerUnit(enum.Enum):
    """Enumeration for the different units a power can be given."""

    W = enum.auto()
    dBW = enum.auto()
    dBm = enum.auto()


# --------------------------------------------------------------------------------------------------
# Signal power calculations
# --------------------------------------------------------------------------------------------------
def free_space_path_loss(freq: float, s_vel: float, dist: float) -> float:
    """Calculate the free space path loss in decibel.

    See wiki Simulation/Signal-Propagation.

    Parameters
    ----------
    freq
        Signal frequency (in Hz).
    s_vel
        Signal velocity (in m/s).
    dist
        Distance (in meter).

    Returns
    -------
    float
        The free space path loss in decibel.
    """
    if dist == 0.0:
        return 0.0

    l_db: float = (
        20.0 * np.log10(4.0 * np.pi / s_vel) + 20.0 * np.log10(freq) + 20.0 * np.log10(dist)
    )

    return l_db


free_space_path_loss_vec = np.vectorize(free_space_path_loss)


def received_power(
    p_s: float,
    g_s_db: float,
    g_r_db: float,
    dist: float,
    freq: float,
    s_vel: float,
    p_unit: PowerUnit,
) -> float:
    """Calculate the received power.

    See wiki Simulation/Signal-Propagation.

    Parameters
    ----------
    p_s
        Power of the sent signal.
    g_s_db
        Sender antenna gain in decibel.
    g_r_db
        Receiver antenna gain in decibel.
    dist
        Distance (in meter).
    freq
        Signal frequency (in Hz).
    s_vel
        Signal velocity (in m/s).
    p_unit
        The power unit of `p_s` and the return value.

    Returns
    -------
    float
        The power at the receiver in the unit specified in `p_unit`.

    Raises
    ------
    ValueError
        If `p_unit` value is invalid.
    """
    l_db = free_space_path_loss(freq=freq, s_vel=s_vel, dist=dist)

    match p_unit:
        case PowerUnit.dBm | PowerUnit.dBW:
            return p_s + g_s_db + g_r_db - l_db
        case PowerUnit.W:
            p_r_dbw: float = 10.0 * np.log10(p_s) + g_s_db + g_r_db - l_db
            p_r_w: float = np.power(10.0, p_r_dbw / 10.0)
            return p_r_w
        case _:
            raise ValueError(f"'p_unit' value {p_unit} is invalid.")


received_power_vec = np.vectorize(received_power, excluded=("p_unit",))


# --------------------------------------------------------------------------------------------------
# Other
# --------------------------------------------------------------------------------------------------
def power_50_ohm_to_vpk(p_dbm: float) -> float:
    """Convert a power in a 50Ω system to a voltage.

    Parameters
    ----------
    p_dbm
        Power in dBm.

    Returns
    -------
    float
        The peak voltage in volt.
    """
    return 10 ** ((p_dbm - 10.0) / 20.0)


power_50_ohm_to_vpk_vec = np.vectorize(power_50_ohm_to_vpk)
