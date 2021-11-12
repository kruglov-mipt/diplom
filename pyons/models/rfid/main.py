from ast import literal_eval as make_tuple
from copy import deepcopy
import click
from dataclasses import dataclass
from typing import Optional, Sequence, List
import multiprocessing
import numpy as np
import pandas as pd
from time import time_ns

import pyons

from pyons.models.rfid.serialization import Input, build_model_descriptor, \
    build_output, pprint_output, Output, parse_tari
from pyons.models.rfid.factory import Factory
from pyons.models.rfid.model import Model
from pyons.models.rfid.generator import Generator
from pyons.models.rfid.journal import Journal
from pyons.models.rfid.protocol import DR, TagEncoding


def simulate(inp: Input, verbose: bool = False) -> Output:
    md = build_model_descriptor(inp)

    if verbose:
        print(f"* Simulating: SimID={inp.simulation_id}, "
            f"V={inp.vehicle_speed}, "
            f"M={md.tag_encoding.name}, "
            f"Tari={md.tari * 1_000_000:g}, "
            f"Q={inp.q}, "
            f"TRExt={inp.trext}, "
            f"TxPower={inp.reader_tx_power}dBm, "
            f"Strategy={md.reader_session_strategy.name}")

    factory = Factory(md)
    model = Model(md)
    pyons.set_model(model)
    model.channel = factory.build_channel()
    model.reader = factory.build_reader(model.channel)
    model.generator = Generator(md)
    pyons.setup_env(log_level=pyons.LogLevel.WARNING)

    journal = Journal()
    journal.channel_state_logging_enabled = False
    journal.inventory_round_logging_enabled = True
    journal.frame_ber_logging_enabled = False
    journal.n_skip_vehicles = 0

    t_start_ns = time_ns()
    pyons.run()
    t_end_ns = time_ns()

    out = build_output(
        journal,
        sim_id=inp.simulation_id,
        elapsed_time=((t_end_ns - t_start_ns) / 1_000_000_000)
    )

    if verbose:
        journal.print_all(
            print_inventory_rounds=False, print_tag_read_data=False,
            print_channel_state=False, print_frame_ber=False
        )
        pprint_output(out)

    return out


def simulate_all(inputs: Sequence[Input],
                 num_proc: int = -1,
                 update_fields: Optional[dict] = None,
                 use_jupyter: bool = False) -> Sequence[Output]:
    t_start = time_ns()
    if num_proc < 0:
        num_proc = max(multiprocessing.cpu_count() - 1, 1)
    if update_fields:
        inputs = list(inputs)
        for i, inp in enumerate(inputs):
            inp: Input = deepcopy(inp)
            for key, value in update_fields.items():
                if key in inp.__dataclass_fields__:
                    setattr(inp, key, value)
            inputs[i] = inp

    # HACK: PyONS has strong intention to use Singleton pattern
    # (for instance, Journal). And very bad reset capabilities.
    # Since then, running models sequentially in the same process
    # leads to errors - some information comes from the previous run.
    # To overcome this issue here, we launch one simulation per worker, and
    # to do this we split inputs by the number of processes in the pool
    # and start a new pool for each new group.
    #
    # Since now each group of N_proc models will start simultaneously,
    # we should take care of time balance between executions:
    # - more cars needed => longer simulation
    # - larger speed => shorter simulation
    # - larger tag data rate => longer simulation
    # To sort, we add an extra parameter "_tag_bitrate" to each parameter
    # and sort 1) by max_vehicles_num (DESC) and 2) by _tag_bitrate (DESC).
    # After simulation, we restore the order.
    for i, inp in enumerate(inputs):
        setattr(inp, '_index', i)
        tari = parse_tari(inp.tari)
        m = TagEncoding.parse(inp.m).value
        trcal = tari * (1 + inp.data0_multiplier) * inp.rtcal_multiplier
        datarate = DR.parse(inp.dr).ratio / (trcal * m)
        setattr(inp, '_tari', tari)
        setattr(inp, '_datarate', datarate)

    max_vehicle_speed = max([inp.vehicle_speed for inp in inputs])

    def get_input_key(inp_: Input):
        return (
            inp_.max_vehicles_num,
            max_vehicle_speed - inp_.vehicle_speed,
            getattr(inp_, '_datarate'),
            25e-6 - getattr(inp_, '_tari'),
        )

    inputs.sort(key=get_input_key, reverse=True)

    # Split inputs into groups by the number of processes.
    i = 0
    inputs_groups_list: List[List[Input]] = []
    while i < len(inputs):
        inputs_groups_list.append(inputs[i:i+num_proc])
        i += num_proc

    # Select the proper tqdm progress bar (different for console and Jupyter).
    if use_jupyter:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Execute the simulations.
    outputs = []
    for inputs_group in tqdm(inputs_groups_list):
        # Require the pool to free process resources after execution
        # by passing maxtaskperchild=1.
        # If used without groups (on the plain inputs list), for some
        # reason reuses the process for the second time.
        with multiprocessing.Pool(num_proc, maxtasksperchild=1) as pool:
            outputs_group = pool.map(simulate, inputs_group)

        # Add index to outputs to sort them back.
        for inp, out in zip(inputs_group, outputs_group):
            setattr(out, '_index', getattr(inp, '_index'))
        outputs.extend(outputs_group)

    # Sort inputs and outputs back using _index attribute.
    inputs.sort(key=lambda inp: getattr(inp, '_index'))
    outputs.sort(key=lambda inp: getattr(inp, '_index'))

    t_elapsed = (time_ns() - t_start) / 1_000_000_000
    print(f"[=] Simulated {len(inputs)} inputs in {t_elapsed} sec.")
    return outputs


def simulate_df(
        df: pd.DataFrame,
        num_proc: int = -1,
        use_jupyter: bool = False,
        update_fields: Optional[dict] = None
) -> pd.DataFrame:

    inputs = df.apply(
        lambda row: Input(**{
            key: value for key, value in row.to_dict().items()
            if key in Input.__dataclass_fields__
        }), axis=1)

    outputs = simulate_all(inputs, num_proc=num_proc, use_jupyter=use_jupyter,
                           update_fields=update_fields)

    for key in Output.__dataclass_fields__:
        if key == 'simulation_id':
            continue
        df[key] = pd.Series([getattr(out, key) for out in outputs])

    return df


@click.group()
def cli():
    pass


@cli.command("csv")
@click.option('-M', '--max-vehicles', type=int, default=None)
@click.option('-j', '--num-cpu', type=int, default=-1)
@click.option('--jupyter', default=False)
@click.argument("file_path", metavar="PATH")
def start_simulate_csv(file_path, jupyter, num_cpu, max_vehicles):
    df = pd.read_csv(file_path, converters={
        'vehicle_direction': make_tuple,
    })

    # Check optional fields those override inputs, and write them to dict
    updates = {}
    if max_vehicles is not None:
        updates['max_vehicles_num'] = max_vehicles

    df = simulate_df(df, num_proc=num_cpu, use_jupyter=jupyter,
                     update_fields=updates)
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    cli()


# OLD UNUSED:
# -----------
# chan_df = journal.list_to_df(journal.Journal().channel_state_journal)
# plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
#                     vehicle_id=1)
# plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
#                     vehicle_id=2)
# plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
#                     vehicle_id=3)
# plot_path_loss_contour(chan_df, 0,
#                        reader_side='front', tag_location='front')
# channel_df = journal.list_to_df(journal.Journal().channel_state_journal)
# fmt_power = '{:.2f}'.format
# fmt_time = '{:.6f}'.format
# fmt_pos = '{:.2f}'.format
# formatters = dict(reader_rx_power=fmt_power, tag_rx_power=fmt_power,
#                   rt_path_loss=fmt_power, tr_path_loss=fmt_power,
#                   channel_time=fmt_time, timestamp=fmt_time,
#                   reader_x=fmt_pos, reader_y=fmt_pos, reader_z=fmt_pos,
#                   tag_x=fmt_pos, tag_y=fmt_pos, tag_z=fmt_pos)
# print(channel_df.to_string(formatters=formatters))


# @pyons.stop_condition()
# def check_sim_time():
#     return pyons.time() > pyons.get_model().params.vehicle_lifetime + 0.1
