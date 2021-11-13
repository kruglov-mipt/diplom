from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from tabulate import tabulate

from pyons.models.rfid.phy import THERMAL_NOISE
from pyons.models.rfid.parameters import ModelDescriptor
from pyons.models.rfid.journal import Journal
from pyons.models.rfid.reader import Reader
from pyons.models.rfid.protocol import Session, Sel, DR, TagEncoding
from pyons.models.rfid.pyradise import parse_antenna_rp, parse_ber_model


@dataclass
class Input:
    # Simulation timings and generators settings
    max_vehicles_num: int = 100
    vehicle_interval_min: float = 0.4
    vehicle_interval_max: float = 0.6
    vehicle_position_update_interval: float = 1e-3
    vehicle_life_time: Optional[float] = None  # if not provided, auto compute

    # Road and vehicles geometry
    num_lanes: int = 2
    lane_width: float = 3.5
    vehicle_speed: float = 60.0  # kmph
    vehicle_length: float = 4.0
    tag_start_offset: float = 31.0
    tag_height: float = 0.5
    vehicle_direction: Tuple[float, float, float] = (1, 0, 0)

    # Antennas positions
    use_tag_front: bool = True
    use_tag_back: bool = True
    use_reader_antenna_front: bool = True
    use_reader_antenna_back: bool = True

    # Channel and radio equipment settings
    use_doppler: bool = True
    ber_model: str = "rayleigh"
    permittivity: float = 15.0
    conductivity: float = 0.03
    thermal_noise: float = THERMAL_NOISE

    # Reader logic settings
    reader_rounds_per_antenna: int = 1
    reader_session_strategy: str = "A"  # "A", "B" or "AB"
    reader_rounds_per_inventory_flag = 1
    m: int = '8'   # 1, 2, 4, 8
    dr: str = "8"  # "8" or "64/3"
    tari: str = '6.25'
    data0_multiplier: float = 2.0
    rtcal_multiplier: float = 2.0
    sl: str = "ALL"
    session: str = '0'
    trext: bool = False
    q: int = 4

    # Reader radio settings, antennas and cables
    reader_antenna_offset: float = 1.0
    reader_antenna_height: float = 5.0
    reader_antenna_rp: str = "dipole"
    reader_antenna_gain: float = 6.0
    reader_antenna_cable_loss: float = -2.0
    reader_antenna_angle: float = np.pi / 4
    reader_antenna_polarization: float = 0.5
    reader_circulator_noise: float = -80.0
    reader_frequency: float = 860_000_000

    # Reader power settings
    reader_tx_power: Union[float, str] = "31.5"
    reader_switch_power: bool = True
    reader_power_on_interval: float = 2.0
    reader_power_off_interval: float = 0.1

    # Tag settings
    epc_size: int = 12
    tid_size: int = 6
    tag_sensitivity: float = -18.0
    tag_antenna_rp: str = "dipole"
    tag_antenna_gain: float = 2.0
    tag_antenna_polarization: float = 1.0
    tag_modulation_loss: float = -12.0
    tag_s1_persistence: float = 0.5
    tag_s2_persistence: float = 2.0
    tag_s3_persistence: float = 2.0
    tag_sl_persistence: float = 2.0


@dataclass
class Output:
    vehicle_read_rate: float
    epc_read_rate: float
    tid_read_rate: float
    avg_antenna_interval: float
    avg_rounds_per_tag: float
    avg_round_duration: float
    avg_num_vehicles_in_round: float
    avg_num_tags_in_round: float
    avg_num_tags_in_busy_round: float
    elapsed_time: float


def build_model_descriptor(inp: Input) -> ModelDescriptor:
    md = ModelDescriptor()
    md.lanes_number = inp.num_lanes

    md.reader_antennas_sides = []
    if inp.use_reader_antenna_front:
        md.reader_antennas_sides.append("front")
    if inp.use_reader_antenna_back:
        md.reader_antennas_sides.append("back")

    md.vehicle_tag_locations = []
    if inp.use_tag_front:
        md.vehicle_tag_locations.append("front")
    if inp.use_tag_back:
        md.vehicle_tag_locations.append("back")

    md.use_doppler = inp.use_doppler
    md.thermal_noise = inp.thermal_noise
    md.permittivity = inp.permittivity
    md.conductivity = inp.conductivity
    md.reader_antenna_angle = inp.reader_antenna_angle
    md.lane_width = inp.lane_width
    md.reader_antenna_offset = inp.reader_antenna_offset
    md.reader_antenna_height = inp.reader_antenna_height
    md.reader_antenna_rp = parse_antenna_rp(inp.reader_antenna_rp)
    md.reader_antenna_gain = inp.reader_antenna_gain
    md.reader_antenna_cable_loss = inp.reader_antenna_cable_loss
    md.reader_antenna_polarization = inp.reader_antenna_polarization
    md.reader_rounds_per_antenna = inp.reader_rounds_per_antenna
    md.reader_frequency = inp.reader_frequency
    md.reader_tx_power = float(inp.reader_tx_power)
    md.reader_circulator_noise = inp.reader_circulator_noise
    md.reader_switch_power = inp.reader_switch_power
    md.reader_power_on_interval = inp.reader_power_on_interval
    md.reader_power_off_interval = inp.reader_power_off_interval
    md.reader_rounds_per_inventory_flag = inp.reader_rounds_per_inventory_flag
    md.reader_session_strategy = \
        Reader.SessionStrategy.parse(inp.reader_session_strategy)
    md.reader_ber_model = parse_ber_model(inp.ber_model)
    md.tari = parse_tari(inp.tari)
    md.data0_multiplier = inp.data0_multiplier
    md.rtcal_multiplier = inp.rtcal_multiplier
    md.sl = Sel.parse(inp.sl)
    md.session = Session.parse(inp.session)
    md.tag_encoding = TagEncoding.parse(inp.m)
    md.dr = DR.parse(inp.dr)
    md.trext = inp.trext
    md.q = inp.q
    md.vehicle_length = inp.vehicle_length
    md.vehicle_speed = inp.vehicle_speed / 3.6
    md.vehicle_position_update_interval = inp.vehicle_position_update_interval
    md.tag_start_offset = inp.tag_start_offset
    md.tag_height = inp.tag_height
    md.vehicle_direction = inp.vehicle_direction
    md.tag_antenna_gain = inp.tag_antenna_gain
    md.tag_antenna_rp = parse_antenna_rp(inp.tag_antenna_rp)
    md.tag_antenna_polarization = inp.tag_antenna_polarization
    md.tag_modulation_loss = inp.tag_modulation_loss
    md.tag_sensitivity = inp.tag_sensitivity
    md.tag_s1_persistence = inp.tag_s1_persistence
    md.tag_s2_persistence = inp.tag_s2_persistence
    md.tag_s3_persistence = inp.tag_s3_persistence
    md.tag_sl_persistence = inp.tag_sl_persistence
    md.epc_size = inp.epc_size
    md.tid_size = inp.tid_size

    if inp.vehicle_life_time is not None:
        md.vehicle_lifetime = inp.vehicle_life_time
    else:
        md.vehicle_lifetime = inp.tag_start_offset * 2 / md.vehicle_speed

    md.vehicle_generation_interval = lambda: np.random.uniform(
        inp.vehicle_interval_min,
        inp.vehicle_interval_max
    )
    md.max_vehicles_num = inp.max_vehicles_num
    return md


def build_output(
        journal: Journal,
        elapsed_time: float = 0.0
) -> Output:
    epc_rate, tid_rate = journal.get_tag_read_rate()
    avg_n_vehicles, avg_n_tags_in_round, avg_n_tags_in_busy_round = \
        journal.get_avg_vehicles_and_tags_num_per_round()

    return Output(
        vehicle_read_rate=journal.get_vehicle_read_rate(),
        epc_read_rate=epc_rate,
        tid_read_rate=tid_rate,
        avg_antenna_interval=journal.get_avg_antenna_interval(),
        avg_rounds_per_tag=journal.get_avg_rounds_per_tag(),
        avg_round_duration=journal.get_avg_round_duration(),
        avg_num_vehicles_in_round=avg_n_vehicles,
        avg_num_tags_in_round=avg_n_tags_in_round,
        avg_num_tags_in_busy_round=avg_n_tags_in_busy_round,
        elapsed_time=elapsed_time
    )


def pprint_output(out: Output):
    rows = [
        ("VEHICLE READ RATE", out.vehicle_read_rate),
        ("TAG EPC READ RATE", out.epc_read_rate),
        ("TAG TID READ RATE", out.tid_read_rate),
        ("AVG NUM VEHICLES PER ROUND", out.avg_num_vehicles_in_round),
        ("AVG NUM TAGS PER ROUND", out.avg_num_tags_in_round),
        ("AVG NUM TAGS PER BUSY ROUND", out.avg_num_tags_in_busy_round),
        ("AVG ROUNDS PER TAG", out.avg_rounds_per_tag),
        ("AVG ANTENNA INTERVAL", out.avg_antenna_interval * 1_000_000),
        ("AVG ROUND DURATION", out.avg_round_duration * 1_000_000),
        ("ELAPSED TIME", out.elapsed_time),
    ]
    print(tabulate(rows))


def parse_tari(s: Union[str, float]) -> float:
    if s == '6.25' or s == 6.25:
        return 6.25e-6
    if s == '12.5' or s == 12.5:
        return 12.5e-6
    if s == '18.75' or s == 18.75:
        return 18.75e-6
    if s == '25' or s == 25:
        return 25e-6
    raise ValueError(f"Unsupprted Tari value '{s}'")
