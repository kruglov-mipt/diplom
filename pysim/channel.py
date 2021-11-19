import numpy as np
import numpy.linalg
import scipy
import scipy.special as special


def to_sin(cos):
    return (1 - cos ** 2) ** .5


def to_log(value, dbm=False, tol=1e-15):
    return 10 * np.log10(value) + 30 * int(dbm) if value >= tol else -np.inf


def from_log(value, dbm=False):
    return 10 ** (value / 10 - 3 * int(dbm))

def vec3D(x, y, z):
    return np.array([x, y, z])

def to_power(value, log=True, dbm=False):
    power = np.abs(value) ** 2
    return to_log(power, dbm=dbm) if log else power


#
# Radiation Pattern
#
def __patch_factor(a_cos, t_cos, wavelen, width, length, tol=1e-9):
    a_sin = to_sin(a_cos)
    t_sin = to_sin(t_cos)
    kw = np.pi / wavelen * width
    kl = np.pi / wavelen * length
    if a_cos < tol:
        return 0
    if np.abs(a_sin) < tol:
        return 1.
    elif np.abs(t_sin) < tol:
        return np.cos(kl * a_sin)
    else:
        return np.sin(kw * a_sin * t_sin) / (kw * a_sin * t_sin) * np.cos(kl * a_sin * t_cos)

def __patch_theta(a_cos, t_cos, wavelen, width, length):
    return __patch_factor(a_cos, t_cos, wavelen, width, length) * t_cos

def __patch_phi(a_cos, t_cos, wavelen, width, length):
    return -1 * __patch_factor(a_cos, t_cos, wavelen, width, length) * to_sin(t_cos) * a_cos

def rp_isotropic(**kwargs):
    return 1.0


def rp_dipole(*, a_cos, tol=1e-9, **kwargs): 
    a_sin = to_sin(a_cos)
    return np.abs(np.cos(np.pi / 2 * a_sin) / a_cos) if a_cos > tol else 0.


def rp_patch(*, a_cos, t_cos, wavelen, width, length, **kwargs):
    return ( np.abs(__patch_factor(a_cos, t_cos, wavelen, width, length)) *
            (t_cos ** 2 + a_cos ** 2 * to_sin(t_cos) ** 2) ** 0.5)

#
# Reflection
#
def __c_parallel(cosine, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    return (eta - cosine ** 2) ** 0.5

def __c_perpendicular(cosine, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    return (eta - cosine ** 2) ** 0.5 / eta

def reflection_constant(**kwargs):
    return -1.0 + 0.j

def reflection(*, cosine, polarization, permittivity, conductivity, wavelen, **kwargs):

    sine = (1 - cosine ** 2) ** .5

    if polarization != 0:
        c_parallel = __c_parallel(cosine, permittivity, conductivity, wavelen)
        r_parallel = (sine - c_parallel) / (sine + c_parallel)
    else:
        r_parallel = 0.j

    if polarization != 1:
        c_perpendicular = __c_perpendicular(cosine, permittivity, conductivity, wavelen)
        r_perpendicular = (sine - c_perpendicular) / (sine + c_perpendicular)
    else:
        r_perpendicular = 0.j

    return polarization * r_parallel + (1 - polarization) * r_perpendicular

#
# Pathloss
#
def two_ray_pathloss(*, time, ground_reflection, wavelen,
                     tx_pos, tx_dir_theta, tx_dir_phi, tx_velocity, tx_rp,
                     rx_pos, rx_dir_theta, rx_dir_phi, rx_velocity, rx_rp, log=False, **kwargs):

    ground_normal = np.array([0, 0, 1])
    rx_pos_refl = np.array([rx_pos[0], rx_pos[1], -rx_pos[2]])  # Reflect RX relatively the ground

    d0_vector = rx_pos - tx_pos
    d1_vector = rx_pos_refl - tx_pos
    d0 = numpy.linalg.norm(d0_vector)
    d1 = numpy.linalg.norm(d1_vector)
    d0_vector_tx_n = d0_vector / d0
    d0_vector_rx_n = -d0_vector_tx_n
    d1_vector_tx_n = d1_vector / d1
    d1_vector_rx_n = np.array([-d1_vector_tx_n[0], -d1_vector_tx_n[1], d1_vector_tx_n[2]])

    # Radiation pattern
    tx_azimuth_0 = np.dot(d0_vector_tx_n, tx_dir_theta)
    rx_azimuth_0 = np.dot(d0_vector_rx_n, rx_dir_theta)
    tx_azimuth_1 = np.dot(d1_vector_tx_n, tx_dir_theta)
    rx_azimuth_1 = np.dot(d1_vector_rx_n, rx_dir_theta)

    tx_tilt_0 = np.dot(d0_vector_tx_n, tx_dir_phi)
    rx_tilt_0 = np.dot(d0_vector_rx_n, rx_dir_phi)
    tx_tilt_1 = np.dot(d1_vector_tx_n, tx_dir_phi)
    rx_tilt_1 = np.dot(d1_vector_rx_n, rx_dir_phi)

    g0 = (tx_rp(a_cos=tx_azimuth_0, t_cos=tx_tilt_0, wavelen=wavelen, **kwargs) *
          rx_rp(a_cos=rx_azimuth_0, t_cos=rx_tilt_0, wavelen=wavelen, **kwargs))

    g1 = (tx_rp(a_cos=tx_azimuth_1, t_cos=tx_tilt_1, wavelen=wavelen, **kwargs) *
          rx_rp(a_cos=rx_azimuth_1, t_cos=rx_tilt_1, wavelen=wavelen, **kwargs))

    # Reflection
    cos_grazing = -1 * np.dot(d1_vector_rx_n, ground_normal)
    r1 = ground_reflection(cosine=cos_grazing, wavelen=wavelen, **kwargs)

    # Doppler's shift
    relative_velocity = rx_velocity - tx_velocity
    velocity_pr_0 = np.dot(d0_vector_tx_n, relative_velocity)
    velocity_pr_1 = np.dot(d1_vector_tx_n, relative_velocity)
    
    k = 2 * np.pi / wavelen

    # print(">>> tx_pos={}, rx_pos={}, k={}, g0={}, d0={}, time={}, velocity_pr_0={}, r1={}, g1={}, "
    #       "d1={}, velocity_pr_1={}".format(tx_pos, rx_pos,
    #       k, g0, d0, time, velocity_pr_0, r1, g1, d1, velocity_pr_1))

    pathloss = .5/k * (     g0 / d0 * np.exp(-1j * k * (d0 - time * velocity_pr_0)) + 
                       r1 * g1 / d1 * np.exp(-1j * k * (d1 - time * velocity_pr_1)) )

    # print(">>> pathloss = {}".format(to_power(pathloss)))
    
    return to_power(pathloss) if log else pathloss

#
# BER computation functions
#
def snr(power, noise):
    return from_log(power - noise)

def snr_full(*, snr, miller=1, symbol=1.25e-6, preamble=9.3e-6, 
                     bandwidth=1.2e6, tol=1e-8, **kwargs):

    if snr < tol:
        return 0.5

    # print(">>> miller={}, symbol={}, preamble={}, bandwidth={}".format(
    #     miller, symbol, preamble, bandwidth
    # ))
    # print('++++++++++', snr, preamble, bandwidth, miller, symbol)
    sync_angle = (snr * preamble * bandwidth) ** -0.5
    return miller * snr * symbol * bandwidth * np.cos(sync_angle) ** 2

def q_func(x):
    return 0.5 - 0.5 * scipy.special.erf(x / 2 ** 0.5)

def ber(snr, distr='rayleigh', tol=1e-8):
    if snr < tol:
        return 0.5

    if distr == 'rayleigh':

        t = (1 + 2 / snr) ** 0.5
        return 0.5 - 1 / t + 2 / np.pi * np.arctan(t) / t

    else:

        t = q_func(snr ** 0.5)
        return 2 * t * (1 - t)

