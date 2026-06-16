"""Per-unit radiometric calibration profile (keyed by USB serial).

A profile is the artifact that makes deploying to many homes a non-event: each
camera is provisioned once (its factory NUC / kt / bt read off the device, see
`provision.py`) into a `<serial>.json`, and any host that opens that camera loads
the matching profile and gets corrected °C with no manual per-site calibration.

Scene parameters (emissivity, reflected temperature, transmission) live in the
profile too so a deployment's defaults travel with it. Until a camera is
provisioned the per-unit fields are absent and the corrector falls back to the
legacy linear decode -- still applying the emissivity correction on top.
"""

import json
import os
from pathlib import Path

import numpy as np

from .corrector import RadiometricCorrector

DEFAULT_DIR = Path(os.environ.get(
    "PYTHERMALCAM_PROFILES",
    Path.home() / ".config" / "pythermalcam" / "profiles"))

# Per-unit factory fields, stored as plain lists in JSON, numpy arrays in memory.
_ARRAY_FIELDS = ("nuc", "kt", "bt")


class Profile:
    """Calibration state for one physical camera.

    Per-unit (from the device): ``nuc`` (NUC table), ``kt``, ``bt``, ``gain_mode``.
    Scene defaults: ``emissivity``, ``t_refl`` (°C), ``tau``, ``use_ems_lut``,
    ``ems_version``. ``serial`` keys the file; ``temp_scale`` is the raw-units/Kelvin
    divisor (16 for the TC002C Duo).
    """

    def __init__(self, serial, model="TC002C-Duo", temp_scale=16.0,
                 emissivity=0.95, t_refl=20.0, tau=1.0,
                 use_ems_lut=False, ems_version=2,
                 nuc=None, kt=None, bt=None, gain_mode=None):
        self.serial = serial
        self.model = model
        self.temp_scale = temp_scale
        self.emissivity = emissivity
        self.t_refl = t_refl
        self.tau = tau
        self.use_ems_lut = use_ems_lut
        self.ems_version = ems_version
        self.nuc = None if nuc is None else np.asarray(nuc, dtype=np.float64)
        self.kt = None if kt is None else np.asarray(kt, dtype=np.float64)
        self.bt = None if bt is None else np.asarray(bt, dtype=np.float64)
        self.gain_mode = gain_mode

    def has_nuc(self):
        """True once the per-unit factory parameters have been provisioned."""
        return all(getattr(self, f) is not None for f in _ARRAY_FIELDS)

    def to_dict(self):
        d = {"serial": self.serial, "model": self.model,
             "temp_scale": self.temp_scale, "emissivity": self.emissivity,
             "t_refl": self.t_refl, "tau": self.tau,
             "use_ems_lut": self.use_ems_lut, "ems_version": self.ems_version,
             "gain_mode": self.gain_mode}
        for f in _ARRAY_FIELDS:
            a = getattr(self, f)
            d[f] = None if a is None else a.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def path_for(cls, serial, directory=DEFAULT_DIR):
        return Path(directory) / f"{serial}.json"

    @classmethod
    def load(cls, serial, directory=DEFAULT_DIR):
        return cls.from_dict(json.loads(cls.path_for(serial, directory).read_text()))

    @classmethod
    def try_load(cls, serial, directory=DEFAULT_DIR):
        """Load the profile for ``serial`` or None if it hasn't been provisioned."""
        p = cls.path_for(serial, directory)
        return cls.from_dict(json.loads(p.read_text())) if p.exists() else None

    def save(self, directory=DEFAULT_DIR):
        path = self.path_for(self.serial, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def corrector(self, tables=None):
        """Build a `RadiometricCorrector` configured from this profile."""
        return RadiometricCorrector(
            emissivity=self.emissivity, t_refl=self.t_refl, tau=self.tau,
            profile=self, temp_scale=self.temp_scale, tables=tables,
            use_ems_lut=self.use_ems_lut, ems_version=self.ems_version)
