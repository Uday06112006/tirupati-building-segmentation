"""
XMP metadata packets carrying drone camera orientation.

EXIF has no standard place for gimbal attitude: it can hold a compass direction
(``GPSImgDirection``) but nothing for pitch or roll. Every photogrammetry engine of
consequence -- WebODM/ODM, Pix4D, Agisoft Metashape, RealityCapture -- therefore reads
orientation from an **XMP** packet instead, chiefly the ``drone-dji`` namespace that
DJI stills carry natively, with the Pix4D ``Camera`` namespace as the vendor-neutral
fallback. We emit both so the frames drop straight into any of them.

Angle convention (DJI's, which the ``drone-dji`` tags define):

* ``GimbalPitchDegree``  0 = horizon, **-90 = straight down (nadir)**
* ``GimbalYawDegree``    degrees clockwise from true north
* ``GimbalRollDegree``   positive = right-hand side of the frame down

``Camera:Yaw/Pitch/Roll`` are written with the *same* values and convention. Note that
interpretation of ``Camera:Pitch`` is not perfectly uniform across vendors -- some
treat 0 as nadir rather than horizon -- so if an engine renders the block upside
down, that sign convention is the first thing to check.
"""

from __future__ import annotations

import logging
from typing import Optional
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger(__name__)

XMP_NS = {
    "drone-dji": "http://www.dji.com/drone-dji/1.0/",
    "Camera": "http://pix4d.com/camera/1.0/",
    "photomechanic": "http://ns.camerabits.com/photomechanic/1.0/",
}

_HEADER = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
_FOOTER = '<?xpacket end="w"?>'


def _signed(value: float, places: int = 2) -> str:
    """DJI writes attitude and altitude with an explicit leading sign."""
    return f"{value:+.{places}f}"


def build_xmp(
    gimbal_yaw: Optional[float] = None,
    gimbal_pitch: Optional[float] = None,
    gimbal_roll: Optional[float] = None,
    absolute_altitude: Optional[float] = None,
    relative_altitude: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    flight_yaw: Optional[float] = None,
    attitude_source: Optional[str] = None,
    software: str = "OrthoGenerator-Microservice/1.0",
) -> bytes:
    """
    Build a complete XMP packet as UTF-8 bytes, ready for a JPEG APP1 segment.

    Only the fields supplied are written; nothing is invented to fill a gap. In
    particular ``FlightRollDegree`` / ``FlightPitchDegree`` are deliberately never
    emitted, because a gimbal decouples camera attitude from airframe attitude -- we
    measure the camera, so claiming to know the airframe would be a fabrication.

    ``attitude_source`` is recorded in a ``photomechanic:Prefs``-style note so a later
    reader can tell derived orientation from telemetry-reported orientation.
    """
    attrs: list[str] = []

    def add(key: str, value: str) -> None:
        attrs.append(f"    {key}={quoteattr(value)}")

    if gimbal_roll is not None:
        add("drone-dji:GimbalRollDegree", _signed(gimbal_roll))
    if gimbal_pitch is not None:
        add("drone-dji:GimbalPitchDegree", _signed(gimbal_pitch))
    if gimbal_yaw is not None:
        add("drone-dji:GimbalYawDegree", _signed(gimbal_yaw % 360.0))
    if flight_yaw is not None:
        add("drone-dji:FlightYawDegree", _signed(flight_yaw % 360.0))
    if absolute_altitude is not None:
        add("drone-dji:AbsoluteAltitude", _signed(absolute_altitude))
    if relative_altitude is not None:
        add("drone-dji:RelativeAltitude", _signed(relative_altitude))
    if latitude is not None:
        add("drone-dji:GpsLatitude", f"{latitude:.8f}")
    if longitude is not None:
        add("drone-dji:GpsLongitude", f"{longitude:.8f}")

    # Vendor-neutral duplicates (Pix4D namespace), same convention as above.
    if gimbal_yaw is not None:
        add("Camera:Yaw", f"{gimbal_yaw % 360.0:.2f}")
    if gimbal_pitch is not None:
        add("Camera:Pitch", f"{gimbal_pitch:.2f}")
    if gimbal_roll is not None:
        add("Camera:Roll", f"{gimbal_roll:.2f}")

    if not attrs:
        logger.debug("build_xmp called with no populated fields; emitting bare packet")

    ns_decl = "\n".join(f'    xmlns:{p}={quoteattr(u)}' for p, u in XMP_NS.items())
    note = ""
    if attitude_source:
        note = (f"\n   <photomechanic:Prefs>{escape(f'attitude-source: {attitude_source}')}"
                f"</photomechanic:Prefs>")

    packet = (
        f'{_HEADER}\n'
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk={quoteattr(software)}>\n'
        f' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about=""\n'
        f'{ns_decl}\n'
        + ("\n".join(attrs) + "\n" if attrs else "")
        + f'  >{note}\n'
        f'  </rdf:Description>\n'
        f' </rdf:RDF>\n'
        f'</x:xmpmeta>\n'
        f'{_FOOTER}'
    )
    return packet.encode("utf-8")
