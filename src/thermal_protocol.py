#!/usr/bin/env python3
"""
thermal_protocol.py - EXPERIMENTAL InfiRay/Topdon vendor command transport.

Reverse-engineered from the Topdon Android app's native libraries
(libircmd.so / libUSBUVCCamera.so). See docs/TC002C-DUO.md for the full writeup.

NOTE: real temperatures already work in thermalcam.py with NO commands at all
(via the 512x484 radiometric mode). This module is NOT needed for that. It exists
for the optional next steps - shutter NUC, full °C calibration, the visible/fused
image - which do need the vendor command channel.

>>> THIS IS UNVERIFIED ON HARDWARE AND NOT WIRED INTO thermalcam.py. <<<
By default it runs DRY (prints the control transfers it WOULD send). Pass --run
to actually talk to the camera, at your own risk.

Safety:
  * Only the safe register reads and the `preview_start` command are implemented.
  * The flash/OEM/erase commands from the SDK are deliberately NOT implemented.
  * A USB replug resets the camera if it ends up in a bad state.

Linux caveat: these are USB *vendor* control transfers (bmRequestType 0x41/0xc1),
not UVC-class controls, so the kernel uvcvideo driver does not expose them. To
send them, libusb must talk to the device directly, which conflicts with
streaming the same interface via v4l2 at the same time. Expect to claim/detach
the interface here and do the video capture through libuvc rather than v4l2 once
this path works. That integration is the open task.
"""

import argparse
import sys
import time

VID, PID = 0x2BDF, 0x0102

# Transport constants (see docs/TC002C-DUO.md).
REQTYPE_WRITE = 0x41   # OUT | vendor | interface
REQTYPE_READ = 0xC1    # IN  | vendor | interface
BREQUEST_WRITE = 0x45  # 'E'
BREQUEST_READ = 0x44   # 'D'
WVALUE = 0x0078
TIMEOUT_MS = 1000

# Registers.
REG_CMD = 0x9D00       # command descriptor
REG_PAYLOAD = 0x9D08   # bulk payload
REG_PARAM = 0x1D08     # command parameters
REG_STATUS = 0x0200    # status poll


class InfiRayDevice:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.dev = None
        self.usb = None

    def open(self):
        import usb.core  # pyusb; only needed for a real run
        self.usb = usb.core
        dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is None:
            raise RuntimeError(f"No {VID:04x}:{PID:04x} device found")
        self.dev = dev
        if not self.dry_run:
            # The kernel uvcvideo driver owns the interface; vendor control
            # transfers need it detached. (This is what stops simultaneous v4l2.)
            for cfg in dev:
                for intf in cfg:
                    n = intf.bInterfaceNumber
                    if dev.is_kernel_driver_active(n):
                        dev.detach_kernel_driver(n)
        return self

    def write_reg(self, reg, data):
        """Vendor WRITE of `data` bytes to register `reg`."""
        data = bytes(data)
        if self.dry_run:
            print(f"  WRITE reg=0x{reg:04x} <- {data.hex(' ')}")
            return len(data)
        return self.dev.ctrl_transfer(REQTYPE_WRITE, BREQUEST_WRITE, WVALUE, reg, data, TIMEOUT_MS)

    def read_reg(self, reg, length):
        """Vendor READ of `length` bytes from register `reg`."""
        if self.dry_run:
            print(f"  READ  reg=0x{reg:04x} ({length} bytes)  [dry-run -> zeros]")
            return bytes(length)
        return bytes(self.dev.ctrl_transfer(REQTYPE_READ, BREQUEST_READ, WVALUE, reg, length, TIMEOUT_MS))

    def poll_status(self, tries=1000):
        """Poll REG_STATUS until the command completes (bit0 clear). Returns the
        last status byte; values > 3 indicate an error per the SDK."""
        for _ in range(tries):
            status = self.read_reg(REG_STATUS, 1)[0]
            if self.dry_run:
                return 0
            if status & 1:        # busy
                continue
            if (status >> 1) & 1 and status > 3:
                raise RuntimeError(f"command error, status=0x{status:02x}")
            return status
        raise TimeoutError("status poll timed out")

    def preview_start(self, width, height, fps=25, mode=0, source=0, path=0):
        """The decoded `preview_start` command (see docs/TC002C-DUO.md).

        NOTE: starts the sensor preview; it does not by itself switch the UVC
        stream to 16-bit. The data-flow-mode command is still to be recovered.
        """
        # 1. command descriptor: opcode 0xc10f, 8-byte arg block.
        self.write_reg(REG_CMD, bytes([0x0F, 0xC1, 0, 0, 0, 0, 0, 0x08]))
        # 2. parameter block (width/height big-endian).
        params = bytes([
            fps & 0xFF,
            0x80 if source == 1 else 0x00,
            (width >> 8) & 0xFF, width & 0xFF,
            (height >> 8) & 0xFF, height & 0xFF,
            mode & 0xFF,
            path & 0xFF,
        ])
        self.write_reg(REG_PARAM, params)
        # 3. wait for completion.
        return self.poll_status()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true",
                   help="Actually send to the camera (default: dry-run, prints only).")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=384)
    args = p.parse_args(argv)

    dev = InfiRayDevice(dry_run=not args.run)
    if args.run:
        try:
            dev.open()
        except Exception as exc:  # pyusb missing, no perms, device busy, ...
            sys.exit(f"open failed: {exc}\n(Try: pip install pyusb; run as root; unplug other users.)")
    print(f"preview_start({args.width}x{args.height}){' [DRY-RUN]' if not args.run else ''}:")
    dev.preview_start(args.width, args.height)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
