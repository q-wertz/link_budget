import marimo

__generated_with = "0.9.28"
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
    return (speed_of_light,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Link budget calculation

        Calculate the link budget based on free-space path loss:

        $$
        \text{FSPL}(\text{dB}) = 20 \log_{10}(d) + 20 \log_{10}(f) + 20 \log_{10} \left( \frac{4 \pi}{c} \right)
        $$

        ## Configuration
        """
    )
    return


@app.cell
def __(mo):
    # Configuration
    ui_signal_freq_mhz = mo.ui.slider(
        start=800.0,
        stop=2000.0,
        step=10.0,
        value=1090.0,
        show_value=True,
        label="Signal frequency [MHz]",
    )
    ui_refractive_index = mo.ui.slider(
        start=1.0,
        stop=3.0,
        step=0.01,
        value=1.0,
        show_value=True,
        label="Refractive index",
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
        options={
            # Type: Gain [dBi]
            "Omnidirectional": 0.0,
            "Dipole antenna": 2.15,
        },
        value="Omnidirectional",
        label="The transmitting antenna type",
    )
    ui_rx_antenna_type = mo.ui.dropdown(
        options={
            # Type: Gain [dBi]
            "Omnidirectional": 0.0,
            "Dipole antenna": 2.15,
        },
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
        ui_refractive_index,
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
    ui_refractive_index,
    ui_rx_antenna_type,
    ui_signal_freq_mhz,
    ui_signal_tx_power,
    ui_tx_antenna_type,
):
    # Calculations
    signal_speed_in_medium = speed_of_light / ui_refractive_index.value
    signal_wavelength = signal_speed_in_medium / (ui_signal_freq_mhz.value * 10**6)

    free_space_path_loss_dB = helpfunctions.free_space_path_loss(
        freq=ui_signal_freq_mhz.value * 10**6,
        s_vel=signal_speed_in_medium,
        dist=ui_distance_km.value * 1000.0,
    )
    free_space_path_loss = (
        (4.0 * np.pi * ui_distance_km.value * 1000.0) / (signal_speed_in_medium)
    ) ** 2

    received_power = helpfunctions.received_power(
        p_s=ui_signal_tx_power.value,
        g_s_db=ui_tx_antenna_type.value,
        g_r_db=ui_rx_antenna_type.value,
        dist=ui_distance_km.value * 1000.0,
        freq=ui_signal_freq_mhz.value * 10**6,
        s_vel=signal_speed_in_medium,
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
        signal_speed_in_medium,
        signal_wavelength,
    )


@app.cell
def __(
    mo,
    signal_wavelength,
    ui_distance_km,
    ui_refractive_index,
    ui_rx_antenna_type,
    ui_signal_freq_mhz,
    ui_tx_antenna_type,
):
    # UI
    mo.vstack(
        [
            mo.hstack([ui_signal_freq_mhz, ui_refractive_index]),
            mo.hstack([mo.md(r"$\Rightarrow$ Wavelength: " + f"{signal_wavelength:.2f}m")]),
            mo.hstack([ui_tx_antenna_type, ui_rx_antenna_type, ui_distance_km]),
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
    mo,
    np,
    pd,
    signal_speed_in_medium,
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
        s_vel=np.full(shape=(len(data_np),), fill_value=signal_speed_in_medium),
        dist=data_np[:, 0] * 1000.0,
    )

    data_np[:, 4] = helpfunctions.received_power_vec(
        p_s=data_np[:, 2],
        g_s_db=np.full(shape=(len(data_np),), fill_value=ui_tx_antenna_type.value),
        g_r_db=np.full(shape=(len(data_np),), fill_value=ui_rx_antenna_type.value),
        dist=data_np[:, 0] * 1000.0,
        freq=np.full(shape=(len(data_np),), fill_value=ui_signal_freq_mhz.value * 10**6),
        s_vel=np.full(shape=(len(data_np),), fill_value=signal_speed_in_medium),
        p_unit=helpfunctions.PowerUnit.dBm,
    )

    data_pd = pd.DataFrame(
        data_np, columns=["distance_km", "tx_power_W", "tx_power_dBm", "fspl_dB", "rx_power_dBm"]
    )

    # TODO: Fix. Calculation is wrong
    mo.ui.table(data_pd)
    return data_np, data_pd, distances, transmission_powers


@app.cell
def __(alt, data_pd, mo):
    rx_power_chart = mo.ui.altair_chart(
        alt.Chart(data_pd)
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
    return (rx_power_chart,)


@app.cell
def __(rx_power_chart):
    rx_power_chart
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Sensing

        ## Configuration
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


if __name__ == "__main__":
    app.run()
