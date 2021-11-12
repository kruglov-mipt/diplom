import matplotlib.pyplot as plt
import numpy as np


def plot_path_loss_contour(df, lane, reader_side, tag_location):
    df = df[(df.reader_lane == df.tag_lane) & (df.reader_lane == lane) &
            (df.reader_side == reader_side) & (df.tag_loc == tag_location) &
            ((reader_side == 'front') & (df.reader_x >= df.tag_x) |
             (reader_side == 'back') & (df.reader_x <= df.tag_x))]
    time = df.channel_time.as_matrix()
    distance = np.abs((df.reader_x - df.tag_x).as_matrix())
    rt_pl = df.rt_path_loss.as_matrix()
    # tr_pl = df.tr_path_loss.as_matrix()
    # ber = df.reader_ber

    tv, dv = np.meshgrid(distance, time)
    _, ax_rt_pl = plt.subplots(1, 1)
    ax_rt_pl.contourf(tv, dv, rt_pl)
    plt.show()


def plot_path_loss_line(chan_df, lane, reader_side, tag_location, vehicle_id):
    df = chan_df
    df = df[(df.reader_lane == df.tag_lane) & (df.reader_lane == lane) &
            (df.reader_side == reader_side) & (df.tag_loc == tag_location) &
            ((reader_side == 'front') & (df.reader_x >= df.tag_x) |
             (reader_side == 'back') & (df.reader_x <= df.tag_x)) &
            (df.vehicle_id == vehicle_id)]

    # print(chan_df.head(10).to_string())
    # print("\n============ PL+BER for lane={}, side={}, location={}, vid={}"
    #       "".format(lane, reader_side, tag_location, vehicle_id))
    # print(df.to_string())
    # print("="*80)

    distance = np.abs((df.reader_x - df.tag_x).as_matrix())
    rt_pl = df.rt_path_loss.as_matrix()
    tr_pl = df.tr_path_loss.as_matrix()
    reader_rx = df.reader_rx_power
    tag_rx = df.tag_rx_power
    ber = df.reader_ber

    _, (ax_power, ax_ber) = plt.subplots(1, 2)
    ax_power.set_title("Power")
    ax_power.set_ylim([-120, 0])
    ax_power.grid()
    ax_power.plot(distance, rt_pl, label='R=>T PL')
    ax_power.plot(distance, tr_pl, label='T=>R PL')
    ax_power.plot(distance, reader_rx, label='Reader RX Power')
    ax_power.plot(distance, tag_rx, label='Tag RX Power')
    ax_power.legend()

    ax_ber.set_title("BER")
    ax_ber.plot(distance, ber)
    ax_ber.set_ylim([0.0, 0.01])
    ax_ber.set_yticks(np.arange(0.0, 0.01, 0.001))
    ax_ber.grid()

    plt.show()
    # print("HO-HO-HO\n" * 10)
    # print(df.to_string())
