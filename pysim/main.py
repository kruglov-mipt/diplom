from dataclasses import dataclass
import itertools
import numpy as np
from multiprocessing import Pool
import pandas
import click
from time import time_ns

from pysim import simulator as sim
from pysim import epcstd as std

import pysim.models as models
from pysim.models import modelParams, KMPH_TO_MPS_MUL


DEFAULT_SPEED = 60             # kmph
DEFAULT_ORIENTATION = 'front'  # front or back
DEFAULT_ENCODING = 'M2'        # FM0, M2, M4, M8
DEFAULT_TARI = "12.5"          # 6.25, 12.5, 18.75, 25
DEFAULT_ANGLE_DEG = 30         # angle in degrees
DEFAULT_USE_TREXT = False      # don't use extended preamble
DEFAULT_USE_DOPPLER = True     # simulate Doppler shift
DEFAULT_FREQUENCY = 860.0      # reader frequency in MHz
DEFAULT_Q = 2                  # Q parameter
DEFAULT_NUM_TAGS = 1_000       # number of tags to simulate

@dataclass
class ModelInput:
    speed: float = DEFAULT_SPEED
    orientation: str = DEFAULT_ORIENTATION
    encoding: str = DEFAULT_ENCODING
    tari: str = DEFAULT_TARI
    angle: float = DEFAULT_ANGLE_DEG
    use_trext: bool = DEFAULT_USE_TREXT
    use_doppler: bool = DEFAULT_USE_DOPPLER
    frequency: float = DEFAULT_FREQUENCY
    q: int = DEFAULT_Q
    num_tags: int = DEFAULT_NUM_TAGS

# ----------------------------------------------------------------------------
@click.group()
def cli():
    pass


@cli.command("start")
@click.option(
    "-s", "--speed", default=DEFAULT_SPEED, show_default=True,
    help="Vehicle speed, kmph",)
@click.option(
    "--front", "orientation", flag_value="front",
    default=(DEFAULT_ORIENTATION == 'front'), help="Use front plates (default)")
@click.option(
    "--back", "orientation", flag_value="back", help="Use back plates",)
@click.option(
    "-m", "--encoding", type=click.Choice(["1", "2", "4", "8"]),
    default=DEFAULT_ENCODING, help="Tag encoding", show_default=True)
@click.option(
    "-t", "--tari", default=DEFAULT_TARI, show_default=True,
    type=click.Choice(["6.25", "12.5", "18.75", "25"]), help="Tari value")
@click.option(
    "-a", "--angle", default=DEFAULT_ANGLE_DEG, show_default=True,
    help="Antenna angle in degrees")
@click.option(
    "--doppler/--no-doppler", "use_doppler", default=DEFAULT_USE_DOPPLER,
    show_default=True, help="Model Doppler shift")
@click.option(
    "--trext/--no-trext", "use_trext", default=DEFAULT_USE_TREXT,
    show_default=True, help="Use extended preamble in tag response")
@click.option(
    "-q", default=DEFAULT_Q, show_default=True, help="Q parameter")
@click.option(
    "--frequency", default=DEFAULT_FREQUENCY, show_default=True,
    help="Reader frequency")
@click.option(
    "-n", "--num-tags", default=DEFAULT_NUM_TAGS, show_default=True,
    help="Number of tags to simulate")
def start_single(**kwargs):
    model_input = ModelInput(**kwargs)
    setup_default_parameters(model_input.num_tags)
    # params = config(BASE_SPEED)

    t_start_ns = time_ns()

    ret = estimate_rates(
        speed=model_input.speed,
        tari=float(model_input.tari) * 1e-6,
        encoding=model_input.encoding,
        orientation=model_input.orientation,
        doppler=model_input.use_doppler,
        trext=model_input.use_trext,
        angle=(model_input.angle * np.pi / 180.0),
        q=model_input.q,
        frequency=(model_input.frequency * 1e6),
    )

    t_end_ns = time_ns()

    print(ret)
    print(f"elapsed: {(t_end_ns - t_start_ns) / 1_000_000_000} sec.")

    # estimate_rates(
    #     speed=d['speed'],
    #     tari=d['tari'],
    #     encoding=d['encoding'],
    #     orientation=d['orientation'],
    #     doppler=d.get('doppler', modelParams.use_doppler),
    #     trext=d.get('trext', modelParams.trext),
    #     angle=d.get('angle', modelParams.reader_antenna_angle),
    #     q=d.get('q', modelParams.q),
    #     frequency=d.get('frequency', modelParams.frequency)
    # )

    # pool = Pool(NUM_THREADS)
    # ret = pool.map(call_estimate_rates, params)
    # df = pandas.DataFrame(ret)
    # df.to_csv("results-3500_60-120-5.csv", index_label='index')


@cli.command("batch")
def start_batch():
    # setup_default_parameters(NUM_TAGS)
    # params = config(BASE_SPEED)
    # print("TOTAL NUMBER OF SETTINGS: {}".format(len(params)))
    # pool = Pool(NUM_THREADS)
    # ret = pool.map(call_estimate_rates, params)
    # df = pandas.DataFrame(ret)
    df.to_csv("results-3500_60-120-5.csv", index_label='index')


# ----------------------------------------------------------------------------



def setup_default_parameters(num_tags):
    modelParams.delim = 12.5e-6  # sec.
    modelParams.tari = 6.25e-6  # sec.
    modelParams.rtcal_tari_mul = 3.0
    modelParams.trcal_rtcal_mul = 2.5
    modelParams.temp = std.TempRange.NOMINAL

    modelParams.speed = 60 * KMPH_TO_MPS_MUL  # meters per second
    modelParams.initial_distance_to_reader = 15.0  # meters
    modelParams.travel_distance = 30.0  # meters
    modelParams.lanes_num = 1  # 1 or 2
    modelParams.lane_width = 3.5  # meters
    modelParams.reader_antenna_sides = ['front', 'back']
    modelParams.reader_antenna_height = 5.0  # meters (z-axis)
    modelParams.reader_antenna_offset = 1.0  # meters (x-axis)
    modelParams.tag_antenna_height = 0.5  # meters (z-axis)
    modelParams.reader_antenna_angle = np.pi / 4.0

    modelParams.tag_orientation = 'front'  # 'front' or 'back'
    modelParams.update_interval = 0.01  # sec.

    # Energy settings
    modelParams.reader_power = 31.5  # dBm
    modelParams.reader_antenna_gain = 6.0  # dB
    modelParams.reader_cable_loss = -2.0  # dB
    modelParams.tag_antenna_gain = 3.0  # dB
    modelParams.tag_modulation_loss = -12.0  # dB
    modelParams.tag_sensitivity = -18.0  # dBm
    modelParams.reader_noise = -80.0  # dBm

    # Medium settings
    modelParams.ber_distribution = 'rayleigh'  # 'rayleigh' or 'awgn'
    modelParams.frequency = 860 * 1e6  # MHz
    modelParams.permittivity = 15.0  # for ground reflection
    modelParams.conductivity = 0.03  # for ground reflection
    modelParams.polarization_loss = -3.0  # dB
    modelParams.ground_reflection_type = 'reflection'  # 'reflection' or 'const'
    modelParams.use_doppler = True

    # Reader power control and antennas switching
    modelParams.reader_switch_power = True
    modelParams.reader_power_on_duration = 2.0  # sec.
    modelParams.reader_power_off_duration = 0.1  # sec.
    modelParams.reader_antenna_switching_interval = 0.1  # sec.
    modelParams.reader_always_start_with_first_antenna = False

    # Inventory settings
    modelParams.read_tid_bank = True
    modelParams.read_tid_words_num = 32
    modelParams.q = 2
    modelParams.tag_encoding = std.TagEncoding.M4
    modelParams.dr = std.DivideRatio.DR_8
    modelParams.sel = std.SelFlag.ALL
    modelParams.session = std.Session.S0
    modelParams.target = std.InventoryFlag.A
    modelParams.trext = True
    modelParams.target_strategy = "const"
    modelParams.rounds_per_target = 1

    # Tag internal settings
    modelParams.epc_bitlen = 96
    modelParams.tid_bitlen = 64
    modelParams.s1_persistence = 2.0  # sec.
    modelParams.s2_persistence = 2.0  # sec.
    modelParams.s3_persistence = 2.0  # sec.

    # Generator settings
    modelParams.generation_interval = (
        lambda: 0.5 + np.random.uniform(-0.1, 0.1),)

    # Statistics
    modelParams.collect_power_statistics = False
    modelParams.tag_orientation = 'front'

    modelParams.reader_antenna_angle = np.pi / 6
    modelParams.reader_antenna_sides = ['front', 'back']
    modelParams.dr = std.DivideRatio.DR_8
    modelParams.q = 2
    modelParams.trext = True
    modelParams.target_strategy = 'switch'
    modelParams.rounds_per_target = 1
    modelParams.max_tags_simulated = num_tags


def parse_tag_encoding(s):
    s = s.upper()
    if s in {'1', "FM0"}:
        return std.TagEncoding.FM0
    elif s in {'2', 'M2'}:
        return std.TagEncoding.M2
    elif s in {'4', 'M4'}:
        return std.TagEncoding.M4
    elif s in {'8', 'M8'}:
        return std.TagEncoding.M8
    else:
        raise ValueError('illegal encoding = {}'.format(s))


def _build_param_mapping(**kwargs):
    speed = kwargs.get('speed', [60.0])
    tari = kwargs.get('tari', [modelParams.tari])
    encoding = kwargs.get('encoding', [modelParams.tag_encoding])
    orientation = kwargs.get('orientation', ['front'])
    doppler = kwargs.get('doppler', [modelParams.use_doppler])
    trext = kwargs.get('trext', [modelParams.trext])
    angle = kwargs.get('angle', [modelParams.reader_antenna_angle])
    q = kwargs.get('q', [modelParams.q])
    frequency = kwargs.get('frequency', [modelParams.frequency])

    tuples = itertools.product(speed, tari, encoding, orientation, doppler,
                               trext, angle, q, frequency)
    return list({
        'speed': t[0], 'tari': t[1], 'encoding': t[2], 'orientation': t[3],
        'doppler': t[4], 'trext': t[5], 'angle': t[6], 'q': t[7],
        'frequency': t[8]} for t in tuples)


def estimate_rates(speed, tari, encoding, orientation,
                   doppler=None, trext=None, angle=None, q=None,
                   frequency=None):
    print("[+] Estimating speed={} kmph, tari={}, m={}, orientation={}, "
          "doppler={}, trext={}, angle={}, q={}, frequency={}"
          "".format(speed, tari, str(encoding), orientation, doppler,
                    trext, angle, q, frequency))
    # If encoding is given as a string, try to parse it (otherwise assume
    # it is given as a TagEncoding value)
    try:
        encoding = parse_tag_encoding(encoding)
    except ValueError:
        pass
    result = models.simulate_tags(
        speed=(speed * KMPH_TO_MPS_MUL), encoding=encoding, tari=tari,
        orientation=orientation, log_level=sim.Logger.Level.WARNING,
        use_doppler=doppler, trext=trext, reader_antenna_angle=angle,
        q=q)
    result['m'] = encoding.name
    result['tari'] = tari
    result['speed'] = speed
    result['orientation'] = orientation
    result['doppler'] = doppler
    result['trext'] = trext
    result['angle'] = angle
    result['q'] = q
    result['frequency'] = frequency
    return result


def call_estimate_rates(d):
    return estimate_rates(
        speed=d['speed'],
        tari=d['tari'],
        encoding=d['encoding'],
        orientation=d['orientation'],
        doppler=d.get('doppler', modelParams.use_doppler),
        trext=d.get('trext', modelParams.trext),
        angle=d.get('angle', modelParams.reader_antenna_angle),
        q=d.get('q', modelParams.q),
        frequency=d.get('frequency', modelParams.frequency)
    )


def config(sb):
    speed = [sb, sb + 5.0, sb + 10.0, sb + 15.0, sb + 20.0, sb + 25.0,
             sb + 30.0, sb + 35.0, sb + 40.0, sb + 45.0, sb + 50.0, sb + 55.0,
             sb + 60.0]
    orientation = ['front', 'back']
    angle = [np.pi / 6]
    q = [2]
    frequency = [860 * 1e6]
    sets = [
        {
            'doppler': [True],
            'trext': [True],
            'tari': [12.5e-6],
            'encoding': ['M2', 'M4', 'M8'],
        }, {
            'doppler': [True],
            'trext': [True],
            'tari': [6.25e-6, 18.75e-6, 25.0e-6],
            'encoding': ['M4'],
        }, {
            'doppler': [False],
            'trext': [True],
            'tari': [12.5e-6],
            'encoding': ['M4'],
        }, {
            'doppler': [True],
            'trext': [False],
            'tari': [12.5e-6],
            'encoding': ['M4'],
        }
    ]
    lists = [_build_param_mapping(
        speed=speed, orientation=orientation, doppler=x['doppler'],
        trext=x['trext'], angle=angle, q=q, frequency=frequency,
        tari=x['tari'], encoding=x['encoding']) for x in sets]
    parameters = []
    for l in lists:
        parameters += l
    return parameters


NUM_TAGS = 3500
NUM_THREADS = 8
BASE_SPEED = 60

if __name__ == '__main__':
    setup_default_parameters(NUM_TAGS)
    params = config(BASE_SPEED)
    print("TOTAL NUMBER OF SETTINGS: {}".format(len(params)))
    pool = Pool(NUM_THREADS)
    ret = pool.map(call_estimate_rates, params)
    df = pandas.DataFrame(ret)
    df.to_csv("results-3500_60-120-5.csv", index_label='index')
