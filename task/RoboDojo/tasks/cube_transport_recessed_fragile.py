"""Move the large cube from its starting fixture into the target zone."""

from .cube_manipulation_geometry import UPRIGHT_TOLERANCE_DEG
from .cube_transport_fixture_common import CubeTransportFixtureCommon


class cube_transport_recessed_fragile(CubeTransportFixtureCommon):
    upright = True
    max_tilt_deg = UPRIGHT_TOLERANCE_DEG
