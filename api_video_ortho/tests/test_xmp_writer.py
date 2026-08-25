"""Tests for the XMP orientation packet and its round-trip through a JPEG."""

import re
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from app.services.xmp_writer import build_xmp, XMP_NS
from app.services.exif_writer import EXIFWriter


def tags(packet: bytes) -> dict:
    text = packet.decode("utf-8")
    return dict(re.findall(r'((?:drone-dji|Camera):\w+)="([^"]+)"', text))


class TestBuildXmp:
    def test_is_well_formed_xml(self):
        packet = build_xmp(gimbal_yaw=153.2, gimbal_pitch=-38.3, gimbal_roll=1.9)
        body = packet.decode("utf-8")
        inner = body[body.index("<x:xmpmeta"):body.index("</x:xmpmeta>") + len("</x:xmpmeta>")]
        ET.fromstring(inner)          # raises if malformed

    def test_has_xpacket_wrapper(self):
        p = build_xmp(gimbal_yaw=1.0).decode("utf-8")
        assert p.startswith("<?xpacket begin=")
        assert p.rstrip().endswith('<?xpacket end="w"?>')

    def test_declares_required_namespaces(self):
        p = build_xmp(gimbal_pitch=-38.0).decode("utf-8")
        for uri in XMP_NS.values():
            assert uri in p

    def test_dji_attitude_tags(self):
        t = tags(build_xmp(gimbal_yaw=153.2, gimbal_pitch=-38.3, gimbal_roll=1.9))
        assert t["drone-dji:GimbalYawDegree"] == "+153.20"
        assert t["drone-dji:GimbalPitchDegree"] == "-38.30"
        assert t["drone-dji:GimbalRollDegree"] == "+1.90"

    def test_pix4d_duplicates_match(self):
        t = tags(build_xmp(gimbal_yaw=153.2, gimbal_pitch=-38.3, gimbal_roll=1.9))
        assert float(t["Camera:Yaw"]) == pytest.approx(153.2)
        assert float(t["Camera:Pitch"]) == pytest.approx(-38.3)
        assert float(t["Camera:Roll"]) == pytest.approx(1.9)

    def test_yaw_normalised_to_0_360(self):
        assert tags(build_xmp(gimbal_yaw=-7.0))["drone-dji:GimbalYawDegree"] == "+353.00"
        assert tags(build_xmp(gimbal_yaw=371.5))["drone-dji:GimbalYawDegree"] == "+11.50"

    def test_nadir_pitch_is_minus_90(self):
        """DJI convention: -90 is straight down, not 0."""
        assert tags(build_xmp(gimbal_pitch=-90.0))["drone-dji:GimbalPitchDegree"] == "-90.00"

    def test_altitudes(self):
        t = tags(build_xmp(absolute_altitude=214.39, relative_altitude=55.19))
        assert t["drone-dji:AbsoluteAltitude"] == "+214.39"
        assert t["drone-dji:RelativeAltitude"] == "+55.19"

    def test_omits_absent_fields(self):
        t = tags(build_xmp(gimbal_pitch=-38.0))
        assert "drone-dji:GimbalPitchDegree" in t
        assert "drone-dji:GimbalYawDegree" not in t
        assert "drone-dji:AbsoluteAltitude" not in t

    def test_never_fabricates_airframe_attitude(self):
        """A gimbal decouples camera from airframe; we measure only the camera."""
        p = build_xmp(gimbal_yaw=153.2, gimbal_pitch=-38.3, gimbal_roll=1.9,
                      flight_yaw=160.5).decode("utf-8")
        assert "FlightRollDegree" not in p
        assert "FlightPitchDegree" not in p
        assert "FlightYawDegree" in p          # course over ground is genuinely known

    def test_attitude_source_recorded(self):
        p = build_xmp(gimbal_pitch=-38.0, attitude_source="visual-homography").decode("utf-8")
        assert "visual-homography" in p

    def test_empty_packet_is_still_valid(self):
        body = build_xmp().decode("utf-8")
        inner = body[body.index("<x:xmpmeta"):body.index("</x:xmpmeta>") + len("</x:xmpmeta>")]
        ET.fromstring(inner)


class TestJpegRoundTrip:
    def test_xmp_survives_save_and_reload(self, tmp_path):
        img = Image.new("RGB", (64, 48), (10, 90, 40))
        exif = EXIFWriter.create_exif_bytes(
            latitude=13.398251, longitude=79.563810, altitude=214.388,
            timestamp_str="2024-11-03 11:35:52", iso=100, shutter="1/500.0",
            fnum=2.8, focal_len=28.0, yaw=152.97, pitch=-37.61, roll=2.09,
            frame_index=241)
        xmp = build_xmp(gimbal_yaw=152.97, gimbal_pitch=-37.61, gimbal_roll=2.09,
                        absolute_altitude=214.388, relative_altitude=55.19,
                        latitude=13.398251, longitude=79.563810)
        out = tmp_path / "f.jpg"
        EXIFWriter.save_image_with_exif(img, str(out), exif, quality=90, xmp_bytes=xmp)

        reloaded = Image.open(out)
        assert "xmp" in reloaded.info
        t = tags(reloaded.info["xmp"])
        assert t["drone-dji:GimbalPitchDegree"] == "-37.61"
        assert t["drone-dji:GimbalRollDegree"] == "+2.09"
        assert t["drone-dji:RelativeAltitude"] == "+55.19"

    def test_exif_and_xmp_coexist(self, tmp_path):
        import piexif
        img = Image.new("RGB", (32, 32))
        exif = EXIFWriter.create_exif_bytes(latitude=13.4, longitude=79.5, altitude=200.0,
                                            yaw=153.0)
        xmp = build_xmp(gimbal_yaw=153.0, gimbal_pitch=-38.0)
        out = tmp_path / "both.jpg"
        EXIFWriter.save_image_with_exif(img, str(out), exif, xmp_bytes=xmp)

        loaded = piexif.load(str(out))
        d = loaded["GPS"][piexif.GPSIFD.GPSImgDirection]
        assert d[0] / d[1] == pytest.approx(153.0, abs=0.01)
        assert "xmp" in Image.open(out).info

    def test_no_xmp_when_not_supplied(self, tmp_path):
        img = Image.new("RGB", (32, 32))
        exif = EXIFWriter.create_exif_bytes(latitude=13.4, longitude=79.5, altitude=200.0)
        out = tmp_path / "plain.jpg"
        EXIFWriter.save_image_with_exif(img, str(out), exif)
        assert "xmp" not in Image.open(out).info
