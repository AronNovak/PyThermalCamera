"""Extract the InfiRay correction lookup tables embedded in libadvirtemp.so.

The Android `libadvirtemp.so` carries the model-wide emissivity / distance /
target-temperature LUTs as plain `.data` arrays of little-endian float64. They are
constants for the sensor family (not per-unit), so we dump them once into
`adv_tables.npz`, which ships with the package; the runtime never needs the .so.

Run once to (re)generate the artifact:

    python -m radiometry.tables extract /path/to/libadvirtemp.so

Everything else loads the shipped npz via `load()`.

Determined by reverse-engineering (see docs/TC002C-DUO.md):
  * arrays are little-endian float64;
  * the 2-D ems correction table is row-major [target_temp][org_ems], i.e. its
    shape is (len(target_temp_list), len(org_ems_list)) -- confirmed both against
    the axis lengths and by which reshape varies smoothly along both axes.
"""

import struct
import sys
from pathlib import Path

import numpy as np

DEFAULT_NPZ = Path(__file__).with_name("adv_tables.npz")

# 2-D ems-correction tables, each reshaped to (target_temp_axis, org_ems_axis).
EMS_TABLES = {
    "ems_correct_table_v1": ("target_temp_list_of_ems_table_v1",
                             "org_ems_list_of_ems_table_v1"),
    "ems_correct_table_v2": ("target_temp_list_of_ems_table_v2",
                             "org_ems_list_of_ems_table_v2"),
}

# 1-D axis / lookup tables to carry through verbatim.
LIST_TABLES = [
    "org_ems_list_of_ems_table_v1", "target_temp_list_of_ems_table_v1",
    "org_ems_list_of_ems_table_v2", "target_temp_list_of_ems_table_v2",
    "dist_table", "new_dist_table", "dist_table_v3",
    "temp_table", "target_temp_table", "new_target_temp_table",
    "target_temp_table_v3",
]


def _read_elf_symbols(path):
    """Minimal ELF64 reader: return (file_bytes, {name: (vma, size, file_off)}).

    Maps every defined symbol's virtual address to a file offset through its own
    section header, so extraction never depends on a hand-computed delta.
    """
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise ValueError(f"{path}: not an ELF64 object")
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        name, typ, flags, addr, off, size, link, info, align, ent = \
            struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        secs.append(dict(addr=addr, off=off, size=size, link=link, type=typ, entsize=ent))
    # Prefer the full symbol table (SHT_SYMTAB=2), fall back to dynsym (=11).
    symsec = next((s for t in (2, 11) for s in secs if s["type"] == t), None)
    if symsec is None:
        raise ValueError(f"{path}: no symbol table")
    strsec = secs[symsec["link"]]

    def cstr(base, rel):
        p = base + rel
        return data[p:data.index(b"\0", p)].decode()

    syms = {}
    for i in range(symsec["size"] // symsec["entsize"]):
        st_name, st_info, st_other, st_shndx, st_value, st_size = \
            struct.unpack_from("<IBBHQQ", data, symsec["off"] + i * symsec["entsize"])
        if st_name == 0 or st_shndx == 0 or st_shndx >= len(secs):
            continue
        sec = secs[st_shndx]
        file_off = st_value - sec["addr"] + sec["off"]
        syms[cstr(strsec["off"], st_name)] = (st_value, st_size, file_off)
    return data, syms


def _grab_f64(data, syms, name):
    vma, size, off = syms[name]
    return np.frombuffer(data[off:off + size], dtype="<f8").astype(np.float64)


def _validate(tables):
    """Guard against a wrong offset/dtype: the axes must be physically sane.

    A misread (e.g. the wrong section delta) lands on neighbouring float64s that
    look emissivity-ish but are not monotonic, so monotonic + range checks catch it.
    """
    for k, lo, hi in [("org_ems_list_of_ems_table_v1", 0.5, 1.0),
                      ("org_ems_list_of_ems_table_v2", 0.5, 1.0),
                      ("target_temp_list_of_ems_table_v1", 200.0, 2000.0),
                      ("target_temp_list_of_ems_table_v2", 200.0, 2000.0),
                      ("dist_table", 0.0, 2000.0),
                      ("temp_table", -60.0, 100.0)]:
        a = tables[k]
        # Non-decreasing (some axes duplicate their terminal value as a clamp),
        # but a wrong offset/dtype still lands on non-monotonic noise -> caught here.
        if not (np.all(np.diff(a) >= 0) and a.max() > a.min()):
            raise ValueError(f"{k}: not monotonic non-decreasing -- bad offset/dtype?")
        if not (lo <= a.min() and a.max() <= hi):
            raise ValueError(f"{k}: out of range [{a.min()},{a.max()}] not in [{lo},{hi}]")
    for name, (taxis, eaxis) in EMS_TABLES.items():
        t = tables[name]
        if t.shape != (tables[taxis].size, tables[eaxis].size):
            raise ValueError(f"{name}: shape {t.shape} != axes "
                             f"({tables[taxis].size},{tables[eaxis].size})")
        if not (0.4 <= t.min() and t.max() <= 1.0):
            raise ValueError(f"{name}: values {t.min()}..{t.max()} outside [0.4,1.0]")


def extract(so_path, out_path=DEFAULT_NPZ):
    """Dump the LUTs from `so_path` (libadvirtemp.so) into `out_path` (.npz)."""
    data, syms = _read_elf_symbols(so_path)
    tables = {n: _grab_f64(data, syms, n) for n in LIST_TABLES}
    for name, (taxis, eaxis) in EMS_TABLES.items():
        flat = _grab_f64(data, syms, name)
        tables[name] = flat.reshape(syms[taxis][1] // 8, syms[eaxis][1] // 8)
    _validate(tables)
    np.savez_compressed(out_path, **tables)
    return tables


def load(path=DEFAULT_NPZ):
    """Load the shipped correction tables as a plain dict of numpy arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "extract":
        out = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NPZ
        t = extract(sys.argv[2], out)
        print(f"Wrote {out} with {len(t)} tables:")
        for k, v in sorted(t.items()):
            print(f"  {k:34} {str(v.shape):12} "
                  f"[{v.min():.4g}, {v.max():.4g}]")
    else:
        print(__doc__)
        sys.exit("usage: python -m radiometry.tables extract <libadvirtemp.so> [out.npz]")
