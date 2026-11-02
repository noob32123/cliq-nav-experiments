"""
Fuzzy directional prompting — computes the target's relative direction
in the drone's body frame and returns a natural language direction phrase.
"""
import math
import numpy as np


def compute_direction(drone_pos, drone_quat_wxyz, target_pos):
    """
    Compute fuzzy direction phrase for AerialVLA prompt.
    Returns a string like "straight ahead " or "to your right ".
    """
    drone_pos = np.array(drone_pos)
    target_pos = np.array(target_pos)
    vec_world = target_pos - drone_pos
    dist_xy = np.sqrt(vec_world[0]**2 + vec_world[1]**2)

    if dist_xy < 0.1:
        return ""

    # Convert world vector to body frame
    w, x, y, z = drone_quat_wxyz
    # Quaternion conjugate (inverse for unit quaternion)
    q_conj = np.array([w, -x, -y, -z])

    def quat_multiply(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ])

    vec_quat = np.array([0, vec_world[0], vec_world[1], vec_world[2]])
    rotated = quat_multiply(quat_multiply(q_conj, vec_quat),
                             np.array([w, x, y, z]))
    body_x, body_y = rotated[1], rotated[2]
    angle_deg = math.degrees(math.atan2(body_y, body_x))

    if -15 < angle_deg <= 15:
        return "straight ahead "
    elif 15 < angle_deg <= 60:
        return "forward-right "
    elif 60 < angle_deg <= 120:
        return "to your right "
    elif 120 < angle_deg <= 180:
        return "to your right rear "
    elif -60 < angle_deg <= -15:
        return "forward-left "
    elif -120 < angle_deg <= -60:
        return "to your left "
    else:
        return "to your left rear "


def build_prompt(direction_text, object_description):
    """Build the AerialVLA prompt: <image> Fly {dir}and find the target. {obj} Action:"""
    return f"<image>\nFly {direction_text}and find the target. {object_description}\nAction:"
