"""Move the large cube off its pinned pedestal into the target zone."""

from .cube_transport_fixture_common import CubeTransportFixtureCommon


class cube_transport_pinned(CubeTransportFixtureCommon):
    auxiliary_label = "retaining_pin"
