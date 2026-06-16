"""Radiometric calibration for InfiRay/Topdon UVC cameras (e.g. TC002C Duo).

Reproduces the camera's factory true-temperature pipeline on Linux/Pi by
reimplementing the math from the Android native libraries in pure numpy. The
native libs are arm64/Bionic/JNI and cannot be loaded off-device, so the model's
correction lookup tables are extracted once (`tables.py` -> `adv_tables.npz`) and
the per-pixel correction is reimplemented (`corrector.py`). Per-unit factory
parameters (NUC / kt / bt) are read off each camera over the USB vendor protocol
and stored in a per-unit profile (`profile.py`).

See docs/TC002C-DUO.md for the reverse-engineering background.
"""
