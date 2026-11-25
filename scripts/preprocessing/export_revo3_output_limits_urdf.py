"""Export the user-selected effort/velocity limits without changing vendor inputs.

The runtime reads the same JSON through actuator_config(); this URDF is a
portable copy, not a replacement for the current authored 144-hull USD.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from viserdex.assets.revo3_spec import REVO3_JOINT_NAMES
from viserdex.supermimic.output_limits import load_output_limits


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'config/revo3_right_limits_reference.urdf'
    target = root / 'config/revo3_right_output_limits.urdf'
    limits = load_output_limits()
    raw = source.read_text()
    parsed = ET.fromstring(raw)
    moving = [j for j in parsed.findall('joint') if j.get('type') != 'fixed']
    if {j.get('name') for j in moving} != set(REVO3_JOINT_NAMES):
        raise ValueError('Reference URDF has unexpected movable joints')
    def update(match):
        tag = re.sub(r'effort="[^"]*"', f'effort="{limits["effort_limit_Nm"]:.17g}"', match[0])
        return re.sub(r'velocity="[^"]*"', f'velocity="{limits["velocity_limit_rad_s"]:.17g}"', tag)
    changed, count = re.subn(r'<limit\b[^>]*>', update, raw)
    if count != 21:
        raise ValueError('Expected exactly 21 joint limit elements')
    actual = ET.fromstring(changed)
    for original, exported in zip(parsed.findall('joint'), actual.findall('joint')):
        if original.get('type') == 'fixed':
            continue
        a, b = original.find('limit'), exported.find('limit')
        assert a.get('lower') == b.get('lower') and a.get('upper') == b.get('upper')
        assert float(b.get('effort')) == limits['effort_limit_Nm']
        assert float(b.get('velocity')) == limits['velocity_limit_rad_s']
    target.write_text(changed)
    print(target)


if __name__ == '__main__':
    main()
