"""
Rule-based collision checker using depth image.
Matches FineCog-Nav paper Section 11.1.1 — checks MOVE_FORWARD, ASCEND, DESCEND.
"""
import numpy as np


class CollisionChecker:
    """
    Checks collision risks from depth image for three action types.
    Uses geometric heuristics (no ML).
    """

    DRONE_WIDTH = 1.0
    DRONE_HEIGHT = 0.1
    FORWARD_THRESH = 5.1
    VERTICAL_THRESH = 2.2
    FOV = 90.0

    def __init__(self):
        pass

    def check(self, depth_img):
        """
        Returns dict of bool warnings for each action type.
        depth_img: numpy array (H, W) or (H, W, 3) — DepthVis is RGB-colorized
        """
        if depth_img is None or depth_img.size == 0:
            return {"move_forward": False, "ascend": False, "descend": False}

        depth = np.array(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]  # use first channel for DepthVis
        h, w = depth.shape

        forward_warn = self._check_forward(depth, h, w)
        ascend_warn, descend_warn = self._check_vertical(depth, h, w)

        return {"move_forward": forward_warn, "ascend": ascend_warn, "descend": descend_warn}

    def _check_forward(self, depth, h, w):
        """Check central region for obstacles closer than threshold."""
        cy, cx = h // 2, w // 2
        half_h = max(1, int(self.DRONE_HEIGHT / 2 / np.tan(np.deg2rad(self.FOV / 2)) * h))
        half_w = max(1, int(self.DRONE_WIDTH / 2 / np.tan(np.deg2rad(self.FOV / 2)) * w))
        y1, y2 = max(0, cy - half_h), min(h, cy + half_h)
        x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
        region = depth[y1:y2, x1:x2]
        valid = region[region > 0]
        if len(valid) > 0 and np.min(valid) < self.FORWARD_THRESH:
            return True
        return False

    def _check_vertical(self, depth, h, w):
        """Check above/below regions for obstacles."""
        ascend_warn = False
        descend_warn = False

        if h < 10:
            return False, False

        # Above region (top portion of depth image → what's above drone)
        top_region = depth[:h // 4, :]
        valid_top = top_region[top_region > 0]
        if len(valid_top) > 0 and np.min(valid_top) < self.VERTICAL_THRESH:
            ascend_warn = True

        # Below region (bottom portion → what's below drone)
        bottom_region = depth[3 * h // 4:, :]
        valid_bottom = bottom_region[bottom_region > 0]
        if len(valid_bottom) > 0 and np.min(valid_bottom) < self.VERTICAL_THRESH:
            descend_warn = True

        return ascend_warn, descend_warn

    def format_warning(self, warnings):
        """Format collision warnings for prompt."""
        fwd = warnings.get("move_forward", False)
        asc = warnings.get("ascend", False)
        desc = warnings.get("descend", False)
        return f"forward blocked:{fwd}, ascend blocked:{asc}, descend blocked:{desc}"
