import marimo

__generated_with = "0.9.30"
app = marimo.App(width="full", app_title="Link budget calculator")


@app.cell
def __():
    import itertools

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from helpfunctions import helpfunctions
    return alt, helpfunctions, itertools, mo, np, pd


@app.cell
def __():
    # Constants
    speed_of_light = 2.99792458e8

    antenna_type_gain: dict[str, float] = {
        # Type:  Gain [dBi]
        "Omnidirectional": 0.0,
        "Hertsche dipole antenna": 1.76,
        "λ/2 dipole (ideal)": 2.15,
        "λ/4 monopole (ideal)": 5.15,
    }
    return antenna_type_gain, speed_of_light


@app.cell
def __(mo):
    mo.md(
        r"""
        # Link budget calculation

        Calculation of the link budget based on free-space path loss:

        $$
        \text{FSPL}[\text{dB}] = 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10} \left( \frac{4 \pi}{c} \right)
        $$

        The power at the receiver $P_r$ can be estimated by

        $$
        \begin{align*}
        P_r[\text{dB}] &= P_s[\text{dB}] + G_s - \text{FSPL}[\text{dB}] + G_r \\
        &= P_s[\text{dB}] + G_s - 20 \log_{10}(d) - 20 \log_{10}(f) - 20 \log_{10} \left( \frac{4 \pi}{c} \right) + G_r 
        \end{align*}
        $$

        where $G_s$ and $G_r$ are the sender and receiver antenna Gains (in dBi).

        ## Configuration
        """
    )
    return


@app.cell
def __(antenna_type_gain, mo):
    # Configuration
    ui_signal_freq_mhz = mo.ui.slider(
        start=800.0,
        stop=2000.0,
        step=10.0,
        value=1090.0,
        show_value=True,
        label="Signal frequency [MHz]",
    )
    ui_signal_tx_power = mo.ui.slider(
        start=100.0,
        stop=300.0,
        step=10.0,
        value=150.0,
        show_value=True,
        label="Signal power [W]",
    )
    ui_tx_antenna_type = mo.ui.dropdown(
        options=antenna_type_gain,
        value="Omnidirectional",
        label="The transmitting antenna type",
    )
    ui_rx_antenna_type = mo.ui.dropdown(
        options=antenna_type_gain,
        value="Omnidirectional",
        label="The receiving antenna type",
    )
    ui_distance_km = mo.ui.slider(
        start=10.0,
        stop=300.0,
        step=10.0,
        value=100.0,
        show_value=True,
        label="Distance [km]",
    )
    return (
        ui_distance_km,
        ui_rx_antenna_type,
        ui_signal_freq_mhz,
        ui_signal_tx_power,
        ui_tx_antenna_type,
    )


@app.cell
def __(mo):
    # TODO: Some "famous" settings that should quickly overwrite the custom chosen values
    ui_presets = mo.ui.dropdown(
        options={
            "ADS-B (1090 MHz)": "adsb_1090",
            "GNSS": "gnss",
            "Custom": "custom",
        },
        value="ADS-B (1090 MHz)",
        label="Select a signal type:",
    )
    return (ui_presets,)


@app.cell
def __(
    helpfunctions,
    np,
    speed_of_light,
    ui_distance_km,
    ui_rx_antenna_type,
    ui_signal_freq_mhz,
    ui_signal_tx_power,
    ui_tx_antenna_type,
):
    # Calculations
    signal_wavelength = speed_of_light / (ui_signal_freq_mhz.value * 10**6)

    free_space_path_loss_dB = helpfunctions.free_space_path_loss(
        freq=ui_signal_freq_mhz.value * 10**6,
        s_vel=speed_of_light,
        dist=ui_distance_km.value * 1000.0,
    )
    free_space_path_loss = (
        (4.0 * np.pi * ui_distance_km.value * 1000.0) / (speed_of_light)
    ) ** 2

    signal_tx_power_dbw = 10 * np.log10(ui_signal_tx_power.value)

    received_power = helpfunctions.received_power(
        p_s=ui_signal_tx_power.value,
        g_s_db=ui_tx_antenna_type.value,
        g_r_db=ui_rx_antenna_type.value,
        dist=ui_distance_km.value * 1000.0,
        freq=ui_signal_freq_mhz.value * 10**6,
        s_vel=speed_of_light,
        p_unit=helpfunctions.PowerUnit.W,
    )
    received_power_dbw: float = 10 * np.log10(received_power)
    received_power_dbm: float = 10 * np.log10(received_power * 1000)
    return (
        free_space_path_loss,
        free_space_path_loss_dB,
        received_power,
        received_power_dbm,
        received_power_dbw,
        signal_tx_power_dbw,
        signal_wavelength,
    )


@app.cell
def __(
    mo,
    signal_tx_power_dbw,
    signal_wavelength,
    ui_distance_km,
    ui_rx_antenna_type,
    ui_signal_freq_mhz,
    ui_signal_tx_power,
    ui_tx_antenna_type,
):
    # UI
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.hstack(
                        [
                            ui_signal_freq_mhz,
                            mo.left(
                                mo.md(r"$\Rightarrow$ Wavelength: " + f"{signal_wavelength:.2f}m")
                            ),
                        ],
                        justify="start",
                    ),
                    mo.hstack(
                        [
                            ui_signal_tx_power,
                            mo.md(
                                r"$\Rightarrow$ Signal power [dBW]: " + f"{signal_tx_power_dbw:.1f}dBW"
                            ),
                        ],
                        justify="start",
                    ),
                ]
            ),
            mo.hstack([ui_tx_antenna_type, ui_rx_antenna_type]),
            ui_distance_km,
        ]
    )
    return


@app.cell
def __(mo):
    mo.md("""## Visualization""")
    return


@app.cell
def __(
    helpfunctions,
    itertools,
    np,
    pd,
    speed_of_light,
    ui_rx_antenna_type,
    ui_signal_freq_mhz,
    ui_signal_tx_power,
    ui_tx_antenna_type,
):
    # Visualization
    distances = np.linspace(0.0, 300.0, 150)
    transmission_powers = (
        ui_signal_tx_power.value - 50.0,
        ui_signal_tx_power.value,
        ui_signal_tx_power.value + 50.0,
    )

    data_np = np.zeros(shape=(len(distances) * len(transmission_powers), 5))
    data_np[:, :2] = np.array([dp for dp in itertools.product(distances, transmission_powers)])
    data_np[:, 2] = 10 * np.log10(data_np[:, 1] * 1000.0)

    data_np[:, 3] = helpfunctions.free_space_path_loss_vec(
        freq=np.full(shape=(len(data_np),), fill_value=ui_signal_freq_mhz.value * 10**6),
        s_vel=np.full(shape=(len(data_np),), fill_value=speed_of_light),
        dist=data_np[:, 0] * 1000.0,
    )

    data_np[:, 4] = helpfunctions.received_power_vec(
        p_s=data_np[:, 2],
        g_s_db=np.full(shape=(len(data_np),), fill_value=ui_tx_antenna_type.value),
        g_r_db=np.full(shape=(len(data_np),), fill_value=ui_rx_antenna_type.value),
        dist=data_np[:, 0] * 1000.0,
        freq=np.full(shape=(len(data_np),), fill_value=ui_signal_freq_mhz.value * 10**6),
        s_vel=np.full(shape=(len(data_np),), fill_value=speed_of_light),
        p_unit=helpfunctions.PowerUnit.dBm,
    )

    data_pd = pd.DataFrame(
        data_np, columns=["distance_km", "tx_power_W", "tx_power_dBm", "fspl_dB", "rx_power_dBm"]
    )

    # TODO: Fix. Calculation is wrong
    # mo.ui.table(data_pd)
    return data_np, data_pd, distances, transmission_powers


@app.cell
def __(
    alt,
    data_pd,
    mo,
    ui_rx_antenna_type,
    ui_signal_tx_power,
    ui_tx_antenna_type,
):
    _rx_power_chart = (
        alt.Chart(
            data_pd,
            title=f"Signal power at a receiver for a transmission power of {ui_signal_tx_power.value}W and total antenna gains of {ui_tx_antenna_type.value + ui_rx_antenna_type.value}dBi",
        )
        .mark_line()
        .encode(
            x=alt.X("distance_km").title("Distance [km]"),
            y=alt.Y(
                "rx_power_dBm",
                scale=alt.Scale(
                    domain=[data_pd["rx_power_dBm"].min() - 10, data_pd["rx_power_dBm"].max() + 30]
                ),
            ).title("RX Power [dBm]"),
            color=alt.Color("tx_power_W:N").title("TX Power [W]"),
        )
    )

    rx_power_chart = mo.ui.altair_chart(_rx_power_chart)
    return (rx_power_chart,)


@app.cell
def __(rx_power_chart):
    rx_power_chart
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Sensing/Visualization

        ## ADC

        The received power at a 50 $\Omega$ system corresponds to a voltage of 

        $$
        U_r = 10^{\frac{P[\text{dBm}] - 10}{20}} \text{V}
        $$

        Depending on the voltage range and resolution of the analog to digital converter (ADC) small voltages might not be "sensed" by the DAC. The the minimum voltage a signal has to trigger at the DAC has to be:

        $$
        \begin{align*}
            U_\text{min} &\geq \frac{U_\text{max}}{2^{N_\text{bit}}} \\
            \Rightarrow{}\quad P_{r,\text{min}} &= 
        \end{align*}
        $$

        where $N_\text{bit}$ is the number of bits of the DAC and $U_\text{max}$ is the maximum allowed voltage of the DAC input.
        """
    )
    return


@app.cell
def __(mo):
    # Configuration
    ui_rx_amplifier_dB = mo.ui.slider(
        start=0,
        stop=60,
        step=1,
        value=32,
        show_value=True,
        label="Amplification at receiver side [dB]",
    )
    ui_adc_n_bits = mo.ui.slider(
        start=7,
        stop=16,
        step=1,
        value=14,
        show_value=True,
        label="Number of bits of ADC",
    )
    ui_adc_v_max = mo.ui.slider(
        start=1.0,
        stop=5.0,
        step=0.1,
        value=1.5,
        show_value=True,
        label="Voltage range of ADC [V]",
    )
    return ui_adc_n_bits, ui_adc_v_max, ui_rx_amplifier_dB


@app.cell
def __(mo, ui_adc_n_bits, ui_adc_v_max, ui_rx_amplifier_dB):
    # UI
    mo.vstack(
        [
            mo.hstack([ui_rx_amplifier_dB, ui_adc_n_bits, ui_adc_v_max]),
        ]
    )
    return


@app.cell
def __(
    helpfunctions,
    received_power_dbm,
    ui_adc_n_bits,
    ui_adc_v_max,
    ui_rx_amplifier_dB,
):
    # Calculations
    tx_voltage = helpfunctions.power_50_ohm_to_vpk(received_power_dbm + ui_rx_amplifier_dB.value)
    tx_voltage_bit_value = tx_voltage * (2**ui_adc_n_bits.value / ui_adc_v_max.value)

    # TODO: Conversion of Receiption power and bits
    return tx_voltage, tx_voltage_bit_value


@app.cell
def __(mo, tx_voltage_bit_value, ui_adc_n_bits):
    def calc_symbol(bit_value: int, n_bits: int) -> str:
        if bit_value > 0.1 * 2**n_bits:
            return "✅"
        elif bit_value > 1:
            return "⚠️"
        else:
            return "❌"


    voltage_warn_symb = calc_symbol(bit_value=tx_voltage_bit_value, n_bits=ui_adc_n_bits.value)

    callout = None
    match voltage_warn_symb:
        case "✅":
            callout = mo.callout(
                value="The signal should be well visible after the ADC.", kind="success"
            )
        case "⚠️":
            callout = mo.callout(
                value="The signal might not be well visable, as it is in the lower 10% of what the ADC can convert.",
                kind="warn",
            )
        case "❌":
            callout = mo.callout(
                value="The signal gets lost in the ADC conversion, as the signal is too weak and cannot be represented.",
                kind="danger",
            )
        case "_":
            mo.MarimoStopError(
                "There is an error in the implementation/the calc_symbol(…) function was changed."
            )
    return calc_symbol, callout, voltage_warn_symb


@app.cell
def __(
    callout,
    free_space_path_loss,
    free_space_path_loss_dB,
    mo,
    received_power,
    received_power_dbm,
    received_power_dbw,
    tx_voltage,
    tx_voltage_bit_value,
    ui_rx_amplifier_dB,
    voltage_warn_symb,
):
    # UI
    mo.vstack(
        [
            mo.md(
                f"""

        |     |       |     |
        | --- | :---: | --- |
        | Free-space path loss      | dB<br>frac         | {free_space_path_loss_dB:.2f} dB<br>{free_space_path_loss:.3} |
        | Received power            | dBW<br>dBm<br>Watt | {received_power_dbw:.2f} dBW<br>{received_power_dbm:.2f} dBm<br>{received_power:.3} W |
        | Amplification in receiver | dB                 | {ui_rx_amplifier_dB.value:.1f} dB |
        | Voltage                   | V                  | {tx_voltage:.2} V |
        | Voltage in bit number     |                    | {tx_voltage_bit_value:.0f} {voltage_warn_symb} |
        """
            ),
            callout,
        ]
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Time domain

        Calculate the number of points a signal has by a set SDR sample rate $f_s$:

        $$
        N = T \cdot f_s
        $$

        where $T$ is the duration of the signal (in s).
        """
    )
    return


@app.cell
def __(mo):
    # Configuration
    ui_sdr_sample_rate = mo.ui.range_slider(
        start=1,
        stop=200,
        step=10,
        value=(10, 120),
        show_value=True,
        label="Sample rate range [MSps]",
    )
    ui_signal_bit_width = mo.ui.slider(
        start=0.1,
        stop=100.0,
        step=0.1,
        value=1.0,
        show_value=True,
        label="Signal length [µs]",
    )
    return ui_sdr_sample_rate, ui_signal_bit_width


@app.cell
def __(mo, ui_sdr_sample_rate, ui_signal_bit_width):
    # UI
    mo.vstack(
        [
            mo.hstack([ui_sdr_sample_rate, ui_signal_bit_width]),
        ]
    )
    return


@app.cell
def __(np, pd, ui_sdr_sample_rate, ui_signal_bit_width):
    # Data generation
    sample_rate_data = pd.DataFrame(
        np.arange(
            start=ui_sdr_sample_rate.value[0],
            stop=ui_sdr_sample_rate.value[1],
            step=1,
        ),
        columns=["sample_rate"],
    )

    sample_rate_data["datapoints"] = (sample_rate_data.loc[:, "sample_rate"] * 1e6) * (
        ui_signal_bit_width.value * 1e-6
    )
    return (sample_rate_data,)


@app.cell
def __(alt, mo, sample_rate_data, ui_signal_bit_width):
    sample_rate_chart = mo.ui.altair_chart(
        alt.Chart(
            sample_rate_data,
            title=f"The number of datapoints a SDR has for a signal with a duration of {ui_signal_bit_width.value:.1f} µs",
        )
        .mark_line()
        .encode(
            x=alt.X("sample_rate").title("Sample rate [MSps]"),
            y=alt.Y("datapoints").title("N datapoints"),
            # color=alt.Color("tx_power_W:N").title("TX Power [W]"),
        )
    )
    return (sample_rate_chart,)


@app.cell
def __(sample_rate_chart):
    sample_rate_chart
    return


if __name__ == "__main__":
    app.run()
