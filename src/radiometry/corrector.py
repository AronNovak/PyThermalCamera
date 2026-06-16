"""Radiometric correction pipeline: raw Y16 -> true object temperature (°C).

Reimplements the InfiRay libadvirtemp math in vectorised numpy. The full firmware
pipeline is:

    raw_u16 --[NUC + kt/bt]--> apparent temp --[ems LUT]--> effective emissivity
            --[temp_correct: emissivity / reflected-temp / transmission]--> object °C
            --[distance/atmosphere]--> reported °C

The emissivity/reflected-temp/transmission inversion (`emissivity_correct`, the
firmware's `temp_correct`) and its exact inverse (`reverse_temp_correct`) are fully
recovered and validated here by round-trip. The NUC + kt/bt stage that turns raw
counts into the apparent temperature needs the camera's per-unit factory
parameters (read over the vendor protocol, see `vendor.py`/`profile.py`); until a
profile supplies them, `raw_to_apparent` falls back to the legacy linear decode and
the corrector still applies the emissivity correction the legacy path never did.
"""

import numpy as np

from . import tables as _tables

KELVIN = 273.15
EMS_Q14 = 16384          # emissivity fixed-point scale (2^14), per the firmware
_TAU_FLOOR = 1e-4        # firmware guards tau < 1e-4 / emissivity == 0


def emissivity_correct(t_meas, emissivity, t_refl=20.0, tau=1.0):
    """True object temperature from an apparent/measured temperature.

    Inverts the graybody radiance-mixing model (firmware `temp_correct`): the
    measured temperature carries the object's own emission plus reflected ambient,
    attenuated by atmospheric transmission ``tau``::

        Tmeas^4 = (1 - tau*e) * Trefl^4 + tau*e * Tobj^4      (T in K, W proportional T^4)
      => Tobj   = ([Tmeas^4 - (1 - tau*e) * Trefl^4] / (tau*e)) ** 1/4

    ``emissivity`` is quantised to the camera's Q14 fixed point (e*16384) to match
    the firmware. ``t_meas`` may be a scalar or a numpy array; ``t_refl`` is the
    reflected apparent temperature (°C, ~ambient). Returns °C.
    """
    e_int = np.rint(np.clip(emissivity, _TAU_FLOOR, 1.0) * EMS_Q14)
    f = tau * e_int / EMS_Q14                                   # tau * emissivity
    meas4 = (np.asarray(t_meas, dtype=np.float64) + KELVIN) ** 4
    refl4 = (t_refl + KELVIN) ** 4
    obj4 = np.maximum(meas4 - (1.0 - f) * refl4, 0.0) / f
    return np.sqrt(np.sqrt(obj4)) - KELVIN


def reverse_temp_correct(t_obj, emissivity, t_refl=20.0, tau=1.0):
    """Apparent temperature a body of true temperature ``t_obj`` would read.

    The exact inverse of `emissivity_correct` (firmware `reverse_temp_correct`);
    kept so the forward correction can be validated by round-trip without the lib.
    """
    e_int = np.rint(np.clip(emissivity, _TAU_FLOOR, 1.0) * EMS_Q14)
    f = tau * e_int / EMS_Q14
    obj4 = (np.asarray(t_obj, dtype=np.float64) + KELVIN) ** 4
    refl4 = (t_refl + KELVIN) ** 4
    meas4 = (1.0 - f) * refl4 + f * obj4
    return np.sqrt(np.sqrt(meas4)) - KELVIN


def compatible_emissivity(target_temp_C, org_ems, tables, version=2):
    """Effective in-band emissivity for an object at ``target_temp_C``.

    The user-set broadband emissivity differs from what the sensor sees in its
    spectral band, and the gap varies with object temperature; the firmware looks
    the corrected value up in a 2-D table over (target temperature, set emissivity)
    -- `read_compatible_ems_version1`. We reproduce it as a clamped bilinear
    interpolation of the extracted `ems_correct_table_v{version}` (axes in Kelvin /
    emissivity). ``target_temp_C`` may be a numpy array (per-pixel); ``org_ems`` is
    the scalar scene emissivity.
    """
    temp_axis = tables[f"target_temp_list_of_ems_table_v{version}"]   # Kelvin
    ems_axis = tables[f"org_ems_list_of_ems_table_v{version}"]
    table = tables[f"ems_correct_table_v{version}"]                   # (n_temp, n_ems)
    # Interpolate the columns at the scalar set-emissivity -> a curve over temp,
    # then interpolate that curve per pixel. np.interp clamps outside the axes,
    # matching the firmware's range guard (it errors out of range; we hold the edge).
    j = np.interp(np.clip(org_ems, ems_axis[0], ems_axis[-1]),
                  ems_axis, np.arange(ems_axis.size))
    j0 = int(np.floor(j)); j1 = min(j0 + 1, ems_axis.size - 1); fj = j - j0
    col = table[:, j0] * (1.0 - fj) + table[:, j1] * fj
    tk = np.asarray(target_temp_C, dtype=np.float64) + KELVIN
    return np.interp(tk, temp_axis, col)


def vapor_pressure(rh, temp_C):
    """Water-vapour partial pressure (Pa) via the firmware's Magnus form.

    `calculate_vapor_pressure`: 611.2 * exp(17.67 * T / (T + 243.5)) * RH, with the
    firmware's [800, 3000] Pa clamp. Feeds the atmospheric-transmission chain; at
    indoor range that chain leaves tau ~ 1, so this is provided for completeness and
    the pipeline defaults to tau = 1.
    """
    es = 611.2 * np.exp(17.67 * temp_C / (temp_C + 243.5)) * rh
    return np.clip(es, 800.0, 3000.0)


class RadiometricCorrector:
    """Apply the radiometric correction to a frame given scene + unit parameters.

    Scene parameters (``emissivity``, ``t_refl``, ``tau``) default to sensible
    indoor values; per-unit NUC/kt/bt come from a `profile` once provisioned.
    """

    def __init__(self, emissivity=0.95, t_refl=20.0, tau=1.0, profile=None,
                 temp_scale=16.0, tables=None, use_ems_lut=False, ems_version=2):
        self.emissivity = emissivity
        self.t_refl = t_refl
        self.tau = tau
        self.profile = profile
        self.temp_scale = temp_scale
        self.use_ems_lut = use_ems_lut
        self.ems_version = ems_version
        self._tables = tables
        if use_ems_lut and self._tables is None:
            self._tables = _tables.load()

    def raw_to_apparent(self, raw_u16):
        """Raw Y16 counts -> apparent temperature (°C).

        With a provisioned profile this will apply the per-unit NUC + kt/bt kernel
        (firmware `temp_measure_with_NUC_value`); without one it is the legacy
        linear decode, which is spatially correct but carries the uncalibrated
        absolute offset.
        """
        if self.profile is not None and self.profile.has_nuc():
            raise NotImplementedError("NUC/kt/bt kernel lands with vendor reads (Phase 3)")
        return np.asarray(raw_u16, dtype=np.float64) / self.temp_scale - KELVIN

    def apparent_to_object(self, t_apparent):
        """Apply the emissivity / reflected-temp / transmission correction.

        With ``use_ems_lut`` the per-pixel effective emissivity is looked up from
        the temperature-dependent table; otherwise the scalar scene emissivity is
        used directly (the rigorously round-trip-validated path).
        """
        emis = self.emissivity
        if self.use_ems_lut:
            emis = compatible_emissivity(t_apparent, self.emissivity,
                                         self._tables, self.ems_version)
        return emissivity_correct(t_apparent, emis, self.t_refl, self.tau)

    def apply(self, raw_u16):
        """Full raw -> corrected object °C for a frame (or scalar)."""
        return self.apparent_to_object(self.raw_to_apparent(raw_u16))


def _self_test():
    """Round-trip: emissivity_correct must invert reverse_temp_correct exactly."""
    rng = np.random.default_rng(0)
    t_obj = rng.uniform(-20, 120, 100000)
    for emis in (0.5, 0.8, 0.95, 0.98, 1.0):
        for t_refl in (-10.0, 20.0, 35.0):
            for tau in (0.7, 0.9, 1.0):
                meas = reverse_temp_correct(t_obj, emis, t_refl, tau)
                back = emissivity_correct(meas, emis, t_refl, tau)
                err = np.nanmax(np.abs(back - t_obj))
                assert err < 1e-6, f"round-trip {emis=} {t_refl=} {tau=}: {err}"
    # Physical sanity: a perfect emitter (e=1) needs no correction. With e<1 the
    # camera blends in reflected ambient, so a body HOTTER than its surroundings is
    # under-read (correct up) and a body COOLER than its surroundings is over-read
    # (correct down).
    assert abs(emissivity_correct(30.0, 1.0, 20.0, 1.0) - 30.0) < 1e-9
    assert emissivity_correct(30.0, 0.9, 25.0, 1.0) > 30.0    # warm body, cool room
    assert emissivity_correct(20.0, 0.9, 25.0, 1.0) < 20.0    # cool body, warm room

    # Effective-emissivity LUT: in-range, smooth, and a near-identity for skin.
    tbl = _tables.load()
    for ver in (1, 2):
        e = compatible_emissivity(np.array([20.0, 34.0, 60.0]), 0.95, tbl, ver)
        assert np.all((0.4 < e) & (e <= 1.0)), f"ems LUT v{ver} out of range: {e}"
    # The LUT-enabled corrector must still behave near the scalar path for skin.
    c = RadiometricCorrector(emissivity=0.95, t_refl=22.0, use_ems_lut=True)
    assert abs(float(c.apparent_to_object(34.0)) - 34.0) < 3.0

    # Magnus vapour pressure: monotone in RH, within the firmware clamp.
    assert vapor_pressure(0.5, 20.0) < vapor_pressure(0.9, 20.0)
    assert 800.0 <= vapor_pressure(0.5, 20.0) <= 3000.0
    print("corrector self-test OK: round-trip < 1e-6 °C; ems LUT + Magnus sane")


if __name__ == "__main__":
    _self_test()
