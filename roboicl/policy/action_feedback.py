"""Proprioceptive tracking receipts, not task progress or grasp verification."""
import math
import numpy as np


def tracking_receipt(target, observed):
    receipt = {}
    for side in ("left", "right"):
        entry = {}
        key = f"{side}_ee_pose"
        if key in target and key in observed:
            desired, actual = (np.asarray(source[key], dtype=float) for source in (target, observed))
            if desired.shape != (7,) or actual.shape != (7,) or not (np.isfinite(desired).all() and np.isfinite(actual).all()):
                raise ValueError("Invalid EE pose in tracking receipt")
            norms = np.linalg.norm(desired[3:]) * np.linalg.norm(actual[3:])
            if norms < 1e-12:
                raise ValueError("Zero quaternion in tracking receipt")
            cosine = np.clip(abs(np.dot(desired[3:], actual[3:])) / norms, 0, 1)
            entry["position_error_m"] = float(np.linalg.norm(desired[:3] - actual[:3]))
            entry["orientation_error_rad"] = float(2 * math.acos(cosine))
        key = f"{side}_arm_joint_state"
        if key in target and key in observed:
            desired, actual = (np.asarray(source[key], dtype=float) for source in (target, observed))
            if desired.shape != (6,) or actual.shape != (6,) or not (np.isfinite(desired).all() and np.isfinite(actual).all()):
                raise ValueError("Invalid joints in tracking receipt")
            entry["max_joint_error_rad"] = float(np.max(np.abs(desired - actual)))
        # ee_joint_state is prev_control normalized target, not measured aperture.
        # Exclude it: comparing two command-side values cannot measure tracking.
        if entry:
            receipt[side] = entry
    return receipt


class TrackingGuard:
    """Interrupt remaining commands after consecutive large arm residuals.

    Experimental generic thresholds; not a collision/grasp detector.
    """
    def __init__(self, position_m=.05, orientation_rad=.5, joint_rad=.5, consecutive=2):
        if min(position_m, orientation_rad, joint_rad) <= 0 or consecutive < 1:
            raise ValueError("Tracking guard thresholds must be positive")
        self.limits = {"position_error_m": position_m, "orientation_error_rad": orientation_rad,
                       "max_joint_error_rad": joint_rad}
        self.required = consecutive
        self.count = 0

    def observe(self, receipt):
        large = any(value > self.limits[key] for arm in receipt.values()
                    for key, value in arm.items() if key in self.limits)
        self.count = self.count + 1 if large else 0
        return self.count >= self.required
