import numpy as np

import pysim.handlers as handlers
from pysim.objects import Reader, Model, Antenna, Generator, Medium
import pysim.epcstd as std
import pysim.simulator as sim


KMPH_TO_MPS_MUL = 1.0 / 3.6


class Settings:
    # PIE settings
    delim = 12.5e-6         # sec.
    tari = 6.25e-6          # sec.
    _rtcal = None           # if None, rtcal_tari_mul is used
    rtcal_tari_mul = 3.0    # used when _rtcal is None, 2.5 <= x <= 3.0
    _trcal = None           # if None, trcal_rtcal_mul is used
    trcal_rtcal_mul = 2.5   # used when _trcal is None, 1.1 <= x <= 3.0
    temp = std.TempRange.NOMINAL

    def get_rtcal(self, tari):
        return tari * self.rtcal_tari_mul

    def get_trcal(self, rtcal):
        return rtcal * self.trcal_rtcal_mul

    @property
    def rtcal(self):
        return self._rtcal if self._rtcal is not None \
            else self.get_rtcal(self.tari)

    @rtcal.setter
    def rtcal(self, value):
        self._rtcal = value

    @property
    def trcal(self):
        return self._trcal if self._trcal is not None \
            else self.get_trcal(self.rtcal)

    @trcal.setter
    def trcal(self, value):
        self._trcal = value

    # Geometry and speed
    speed = 60 * KMPH_TO_MPS_MUL                # meters per second
    initial_distance_to_reader = 15.0           # meters
    travel_distance = 30.0                      # meters
    lanes_num = 1                               # 1 or 2
    lane_width = 3.5                            # meters
    reader_antenna_sides = ['front', 'back']
    reader_antenna_height = 5.0                 # meters (z-axis)
    reader_antenna_offset = 1.0                 # meters (x-axis)
    tag_antenna_height = 0.5                    # meters (z-axis)

    tag_orientation = 'front'                   # 'front' or 'back'
    update_interval = 0.01                      # sec.

    # Energy settings
    reader_power = 31.5                         # dBm
    reader_antenna_gain = 6.0                   # dB
    reader_cable_loss = -2.0                    # dB
    tag_antenna_gain = 3.0                      # dB
    tag_modulation_loss = -12.0                 # dB
    tag_sensitivity = -18.0                     # dBm
    reader_noise = -80.0                        # dBm

    # Medium settings
    ber_distribution = 'rayleigh'               # 'rayleigh' or 'awgn'
    frequency = 860 * 1e6                       # MHz
    permittivity = 15.0                         # for ground reflection
    conductivity = 0.03                         # for ground reflection
    polarization_loss = -3.0                    # dB
    ground_reflection_type = 'reflection'       # 'reflection' or 'const'
    use_doppler = True

    # Reader power control and antennas switching
    reader_switch_power = True
    reader_power_on_duration = 2.0              # sec.
    reader_power_off_duration = 0.1             # sec.
    reader_antenna_switching_interval = 0.1     # sec.
    reader_always_start_with_first_antenna = False
    reader_antenna_angle = np.pi / 4.0

    # Inventory settings
    read_tid_bank = True
    read_tid_words_num = 4  # was: 32 (?!)
    q = 2
    tag_encoding = std.TagEncoding.M4
    dr = std.DivideRatio.DR_8
    sel = std.SelFlag.ALL
    session = std.Session.S0
    target = std.InventoryFlag.A
    trext = True
    target_strategy = "const"
    rounds_per_target = 1

    # Tag internal settings
    epc_bitlen = 96
    tid_bitlen = 64
    s1_persistence = 2.0                        # sec.
    s2_persistence = 2.0                        # sec.
    s3_persistence = 2.0                        # sec.

    # Generator settings
    generation_interval = (lambda: 1.0, )
    max_tags_simulated = 5

    # Statistics
    collect_power_statistics = False

    def get_power_control_mode(self, reader_switch_power=None):
        x = reader_switch_power if reader_switch_power is not None \
            else self.reader_switch_power
        return Reader.PowerControlMode.PERIODIC if x else \
            Reader.PowerControlMode.ALWAYS_ON


modelParams = Settings()


def simulate_tags(settings=None, speed=None, encoding=None, tari=None,
                  dr=None, trext=None, q=None, session=None,
                  target=None, target_strategy=None, rounds_per_target=None,
                  antenna_switching_interval=None, orientation=None,
                  reader_antenna_angle=None, sim_time_limit=None,
                  real_time_limit=None, log_level=sim.Logger.Level.INFO,
                  generation_const_interval=None, tags_num=None,
                  use_doppler=None, frequency=None):
    settings = settings if settings is not None else modelParams

    # 0) Building the model

    model = Model()
    model.max_tags_num = tags_num if tags_num is not None \
        else settings.max_tags_simulated
    model.update_interval = settings.update_interval
    model.statistics.use_power_statistics = settings.collect_power_statistics

    # 1) Building the reader

    reader = Reader()
    model.reader = reader

    reader.tari = tari if tari is not None else settings.tari
    reader.rtcal = settings.get_rtcal(reader.tari)
    reader.trcal = settings.get_trcal(reader.rtcal)
    reader.delim = settings.delim
    reader.temp = settings.temp
    reader.q = q if q is not None else settings.q
    reader.session = session if session is not None else settings.session
    reader.target = target if target is not None else settings.target
    reader.sel = settings.sel
    reader.dr = dr if dr is not None else settings.dr
    reader.trext = trext if trext is not None else settings.trext
    reader.tag_encoding = encoding if encoding is not None \
        else settings.tag_encoding
    reader.target_strategy = target_strategy if target_strategy is not None \
        else settings.target_strategy
    reader.rounds_per_target = \
        rounds_per_target if rounds_per_target is not None \
        else settings.rounds_per_target
    reader.power_control_mode = settings.get_power_control_mode()
    reader.max_power = settings.reader_power
    reader.power_on_duration = settings.reader_power_on_duration
    reader.power_off_duration = settings.reader_power_off_duration
    reader.noise = settings.reader_noise
    reader.read_tid_bank = settings.read_tid_bank
    reader.read_tid_words_num = settings.read_tid_words_num
    reader.always_start_with_first_antenna = \
        settings.reader_always_start_with_first_antenna
    reader.antenna_switch_interval = antenna_switching_interval \
        if antenna_switching_interval is not None \
        else settings.reader_antenna_switching_interval

    # 2) Attaching antennas to reader
    lane_centers = []
    if settings.lanes_num == 1:
        lane_centers.append(0.0)
    elif settings.lanes_num == 2:
        y = settings.lane_width / 2
        lane_centers.append(-y)
        lane_centers.append(y)
    else:
        raise ValueError("support only 1 or 2 lanes")
    if reader_antenna_angle is None:
        reader_antenna_angle = settings.reader_antenna_angle
    for y in lane_centers:
        for side in settings.reader_antenna_sides:
            ant = Antenna()
            if side == 'front':
                x = settings.reader_antenna_offset
                dx = np.sin(reader_antenna_angle)
            elif side == 'back':
                x = -1.0 * settings.reader_antenna_offset
                dx = -np.sin(reader_antenna_angle)
            else:
                raise ValueError(
                    "unsupported reader antenna side '{}'".format(side))
            z = settings.reader_antenna_height
            dy = 0.0
            dz = -np.cos(reader_antenna_angle)
            ant.pos = np.asarray([x, y, z])
            ant.direction_theta = np.asarray([dx, dy, dz])
            ant.gain = settings.reader_antenna_gain
            ant.cable_loss = settings.reader_cable_loss
            reader.attach_antenna(ant)

    # 3) Setting up medium
    medium = Medium()
    model.medium = medium

    medium.ber_distribution = settings.ber_distribution
    medium.ground_reflection_type = settings.ground_reflection_type
    medium.frequency = frequency if frequency is not None else \
        settings.frequency
    medium.permittivity = settings.permittivity
    medium.conductivity = settings.conductivity
    medium.polarization_loss = settings.polarization_loss
    medium.use_doppler = use_doppler if use_doppler is not None else \
        settings.use_doppler

    # 4) Generator settings
    tag_x0 = settings.initial_distance_to_reader
    tag_z0 = settings.tag_antenna_height
    prefixes = ['A', 'B']
    for tag_y0 in lane_centers:
        generator = Generator()
        model.generators.append(generator)

        generator.pos0 = np.asarray([tag_x0, tag_y0, tag_z0])
        generator.velocity = speed if speed is not None else settings.speed
        generator.direction = np.asarray([-1, 0, 0])
        generator.travel_distance = settings.travel_distance

        prefix = prefixes.pop(0)
        generator.epc_prefix = prefix * 4
        generator.tid_prefix = prefix * 4
        generator.epc_bitlen = settings.epc_bitlen
        generator.tid_bitlen = settings.tid_bitlen

        generator.max_tags_generated = model.max_tags_num
        generator.antenna_gain = settings.tag_antenna_gain
        generator.modulation_loss = settings.tag_modulation_loss
        generator.sensitivity = settings.tag_sensitivity

        orientation = orientation if orientation is not None \
            else settings.tag_orientation
        generator.inversed_antenna_direction = orientation == 'back'

        if generation_const_interval is not None:
            generator.set_interval(lambda: generation_const_interval)
        else:
            generator.set_interval(settings.generation_interval[0],
                                   *settings.generation_interval[1:])

    # 5) Launching simulation
    kernel = sim.Kernel()
    kernel.max_simulation_time = sim_time_limit
    kernel.max_real_time = real_time_limit
    kernel.context = model
    kernel.logger.level = log_level
    kernel.run(handlers.start_simulation)

    return {
        'rounds_per_tag': model.statistics.average_rounds_per_tag(),
        'inventory_prob': model.statistics.inventory_probability(),
        'read_tid_prob': model.statistics.read_tid_probability()
    }
