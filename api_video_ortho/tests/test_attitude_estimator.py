"""Unit tests for attitude recovery geometry (no video decoding required)."""

import math

import numpy as np
import pytest

from app.services.attitude_estimator import (
    compute_track_dynamics,
    focal_px_from_35mm,
    geodetic_to_enu,
    meters_per_degree,
    pitch_roll_from_down_vector,
    yaw_from_motion,
    _mad_inliers,
)


def down_vector(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """World-'down' in camera axes for a given pitch/roll (inverse of the estimator)."""
    tilt = math.radians(90.0 + pitch_deg)   # angle between optical axis and straight down
    r = math.radians(roll_deg)
    return np.array([math.sin(tilt) * math.sin(r),
                     math.sin(tilt) * math.cos(r),
                     math.cos(tilt)])


class TestPitchRoll:
    def test_nadir(self):
        pitch, roll = pitch_roll_from_down_vector([0.0, 0.0, 1.0])
        assert pitch == pytest.approx(-90.0, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)

    def test_horizontal(self):
        pitch, roll = pitch_roll_from_down_vector([0.0, 1.0, 0.0])
        assert pitch == pytest.approx(0.0, abs=1e-6)
        assert roll == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("p,r", [(-38.0, 1.5), (-90.0, 0.0), (-45.0, 10.0),
                                     (-20.0, -7.5), (-75.0, 3.0)])
    def test_roundtrip(self, p, r):
        pitch, roll = pitch_roll_from_down_vector(down_vector(p, r))
        assert pitch == pytest.approx(p, abs=1e-6)
        assert roll == pytest.approx(r, abs=1e-6)

    def test_does_not_mutate_input(self):
        d = np.array([0.0, 2.0, 2.0])          # deliberately not unit length
        before = d.copy()
        pitch_roll_from_down_vector(d)
        np.testing.assert_array_equal(d, before)


class TestYaw:
    @pytest.mark.parametrize("bearing", [0.0, 90.0, 160.5, 275.0, 359.0])
    def test_camera_aligned_with_track(self, bearing):
        """Camera looking along the direction of travel => yaw equals the bearing."""
        pitch = -45.0
        d = down_vector(pitch, 0.0)
        tilt = math.radians(90.0 + pitch)
        # horizontal forward direction in camera axes, orthogonal to `d`
        motion = np.array([0.0, -math.cos(tilt), math.sin(tilt)])
        assert abs(float(np.dot(d, motion))) < 1e-9
        assert yaw_from_motion(d, motion, bearing) == pytest.approx(bearing, abs=1e-6)

    def test_yaw_offset_from_track(self):
        """Rotating the camera about gravity offsets yaw by the same angle."""
        pitch, offset, bearing = -38.0, -7.0, 160.0
        d = down_vector(pitch, 0.0)
        tilt = math.radians(90.0 + pitch)
        fwd = np.array([0.0, -math.cos(tilt), math.sin(tilt)])
        # travel 'offset' degrees to the right of where the camera points
        a = math.radians(offset)
        right = np.cross(d, fwd)
        motion = math.cos(a) * fwd - math.sin(a) * right
        yaw = yaw_from_motion(d, motion, bearing)
        assert (yaw - bearing + 540) % 360 - 180 == pytest.approx(offset, abs=1e-6)

    def test_unobservable_when_motion_is_vertical(self):
        d = np.array([0.0, 0.0, 1.0])
        assert math.isnan(yaw_from_motion(d, d, 90.0))

    def test_does_not_mutate_inputs(self):
        d = down_vector(-38.0, 0.0)
        m = np.array([0.0, -0.5, 0.5])
        db, mb = d.copy(), m.copy()
        yaw_from_motion(d, m, 160.0)
        np.testing.assert_array_equal(d, db)
        np.testing.assert_array_equal(m, mb)


class TestTrackDynamics:
    @staticmethod
    def straight_track(bearing_deg, speed, n=200, fps=24.0, lat0=13.4, lon0=79.56,
                       quantise=True):
        t = np.arange(n) / fps
        m_lat, m_lon = meters_per_degree(lat0)
        e = speed * math.sin(math.radians(bearing_deg)) * t
        nn = speed * math.cos(math.radians(bearing_deg)) * t
        lat = lat0 + nn / m_lat
        lon = lon0 + e / m_lon
        if quantise:                       # DJI SRT reports 6 decimal places
            lat, lon = np.round(lat, 6), np.round(lon, 6)
        return lat, lon, np.full(n, 100.0), t

    @pytest.mark.parametrize("bearing", [0.0, 45.0, 160.5, 250.0, 330.0])
    def test_course_survives_coordinate_quantisation(self, bearing):
        """The headline claim: windowed fitting beats 0.11 m lat/lon quantisation."""
        lat, lon, alt, t = self.straight_track(bearing, 9.0)
        dyn = compute_track_dynamics(lat, lon, alt, t)
        c = dyn.course_deg[~np.isnan(dyn.course_deg)]
        assert len(c) > 150
        err = (c - bearing + 540) % 360 - 180
        assert np.abs(err).max() < 2.0
        assert np.abs(err).mean() < 0.6

    def test_naive_differencing_would_fail(self):
        """Consecutive-frame atan2 on the same track is an order of magnitude worse."""
        bearing = 160.5
        lat, lon, alt, t = self.straight_track(bearing, 9.0)
        e, n, _ = geodetic_to_enu(lat, lon, alt)
        de, dn = np.diff(e), np.diff(n)
        moved = np.hypot(de, dn) > 1e-9
        naive = np.degrees(np.arctan2(de[moved], dn[moved]))
        naive_err = np.abs((naive - bearing + 540) % 360 - 180)
        dyn = compute_track_dynamics(lat, lon, alt, t)
        c = dyn.course_deg[~np.isnan(dyn.course_deg)]
        windowed_err = np.abs((c - bearing + 540) % 360 - 180)
        assert windowed_err.mean() < naive_err.mean() / 5.0

    def test_ground_speed(self):
        lat, lon, alt, t = self.straight_track(90.0, 7.5)
        dyn = compute_track_dynamics(lat, lon, alt, t)
        assert dyn.ground_speed[20:-20].mean() == pytest.approx(7.5, rel=0.05)

    def test_stationary_course_is_nan(self):
        n = 120
        t = np.arange(n) / 24.0
        lat = np.full(n, 13.4)
        lon = np.full(n, 79.56)
        dyn = compute_track_dynamics(lat, lon, np.full(n, 100.0), t)
        assert np.isnan(dyn.course_deg).all()
        assert (dyn.course_quality < 1e-6).all()

    def test_vertical_speed(self):
        n = 120
        t = np.arange(n) / 24.0
        alt = 100.0 + 2.0 * t                      # climbing 2 m/s
        lat, lon, _, _ = self.straight_track(0.0, 5.0, n=n)
        dyn = compute_track_dynamics(lat, lon, alt, t)
        assert dyn.vertical_speed[20:-20].mean() == pytest.approx(2.0, rel=0.05)


class TestIntrinsics:
    def test_focal_px_from_35mm(self):
        # 28 mm equivalent across a 3840 px frame -> ~2987 px
        assert focal_px_from_35mm(28.0, 3840) == pytest.approx(2986.7, abs=1.0)

    def test_wider_lens_gives_shorter_focal(self):
        assert focal_px_from_35mm(16.0, 3840) < focal_px_from_35mm(50.0, 3840)


class TestAttitudeTrack:
    @staticmethod
    def sample_frames():
        from app.services.attitude_estimator import FrameAttitude
        return [FrameAttitude(
            frame_id=i, time_sec=i / 24.0, timestamp="2024-11-03 11:35:42.416",
            latitude=13.4, longitude=79.56, altitude=214.0,
            estimated_yaw=160.5, ground_speed=9.0, heading_change=0.1,
            vertical_speed=-0.1, camera_yaw=153.0 + i * 0.01, camera_pitch=-38.0,
            camera_roll=1.5, agl_m=55.0, attitude_source="visual-homography",
            course_quality=1.0) for i in range(1, 21)]

    def test_lookup_and_len(self):
        from app.services.attitude_estimator import AttitudeTrack
        tr = AttitudeTrack(self.sample_frames())
        assert len(tr) == 20
        assert tr.solved_count == 20
        assert tr.get(5).camera_yaw == pytest.approx(153.05)

    def test_nearest_fallback_for_unsolved_frame(self):
        from app.services.attitude_estimator import AttitudeTrack
        tr = AttitudeTrack(self.sample_frames())
        assert tr.get(999) is tr.get(20)
        assert tr.get(-5) is tr.get(1)

    def test_empty_track_returns_none(self):
        from app.services.attitude_estimator import AttitudeTrack
        tr = AttitudeTrack([])
        assert len(tr) == 0
        assert tr.get(1) is None

    def test_csv_roundtrip(self, tmp_path):
        from app.services.attitude_estimator import AttitudeTrack, write_csv
        frames = self.sample_frames()
        p = tmp_path / "att.csv"
        write_csv(frames, str(p))
        tr = AttitudeTrack.from_csv(str(p))
        assert len(tr) == 20
        got, want = tr.get(7), frames[6]
        assert got.camera_pitch == pytest.approx(want.camera_pitch)
        assert got.camera_yaw == pytest.approx(want.camera_yaw)
        assert got.agl_m == pytest.approx(want.agl_m)
        assert got.attitude_source == want.attitude_source

    def test_csv_roundtrip_preserves_blanks(self, tmp_path):
        from app.services.attitude_estimator import AttitudeTrack, FrameAttitude, write_csv
        f = FrameAttitude(frame_id=1, time_sec=0.0, timestamp=None, latitude=13.4,
                          longitude=79.5, altitude=200.0, estimated_yaw=None,
                          ground_speed=0.0, heading_change=None, vertical_speed=0.0)
        p = tmp_path / "blank.csv"
        write_csv([f], str(p))
        got = AttitudeTrack.from_csv(str(p)).get(1)
        assert got.camera_pitch is None
        assert got.estimated_yaw is None
        assert got.attitude_source == "none"


class TestOutlierRejection:
    def test_rejects_gross_outlier(self):
        v = np.array([1.5, 1.6, 1.4, 1.5, 1.55, 24.4, 1.45, 1.5])
        keep = _mad_inliers(v)
        assert not keep[5]
        assert keep.sum() == 7

    def test_keeps_all_when_consistent(self):
        assert _mad_inliers(np.array([1.5, 1.6, 1.4, 1.5, 1.55, 1.45])).all()

    def test_handles_nan(self):
        keep = _mad_inliers(np.array([1.5, np.nan, 1.4, 1.5, 1.55, 1.45]))
        assert not keep[1]
