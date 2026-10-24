"""Labelled kinematic tree of the duck. Each joint owns a `<joint>_stage` group
holding its horn, the links it drives and the servo bodies mounted on them."""
from __future__ import annotations

from cadgen import build123d as bd, srgb

from lib import parts
from lib.dims import JOINTS

YELLOW = srgb("#F7C531")
ORANGE = srgb("#F08A24")
FRAME = srgb("#4A5560")
SERVO = srgb("#2B2F33")
HORN = srgb("#B8C0C8")
ELEC = srgb("#2E6B4F")
BATT = srgb("#3A5BA0")
GLASS = srgb("#A8D8F0", 0.55)
LCD = srgb("#101418")
PAD = srgb("#3C3C3C")


def _p(shape, label, color):
    shape.label = label
    shape.color = color
    return shape


def stage_parts():
    """Parts owned by each joint (None = body root), in world zero pose."""
    P = {None: [
        _p(parts.body_shell(True), "body_shell_upper", YELLOW),
        _p(parts.body_shell(False), "body_shell_lower", YELLOW),
        _p(parts.wing(1), "wing_l", YELLOW), _p(parts.wing(-1), "wing_r", YELLOW),
        _p(parts.tail(), "tail", YELLOW),
        _p(parts.pelvis_frame(), "pelvis_frame", FRAME),
        _p(parts.battery(), "battery", BATT),
        _p(parts.board(), "rk3566_board", ELEC),
        _p(parts.speaker(), "speaker", SERVO),
        _p(parts.tof_sensor(), "tof_sensor", ELEC),
    ]}
    for name in JOINTS:
        P[name] = [_p(parts.servo_horn(name), f"horn_{name}", HORN)]
    for name, (parent, *_rest) in JOINTS.items():
        P[parent].append(_p(parts.servo_body(name), f"servo_{name}", SERVO))
    for side, s in (("l", 1), ("r", -1)):
        P[f"{side}_hip_yaw"].append(_p(parts.yaw_bracket(s), f"yaw_bracket_{side}", FRAME))
        P[f"{side}_hip_roll"].append(_p(parts.hip_bracket(s), f"hip_bracket_{side}", FRAME))
        P[f"{side}_hip_pitch"].append(_p(parts.thigh(s), f"thigh_{side}", FRAME))
        P[f"{side}_knee"].append(_p(parts.shin(s), f"shin_{side}", FRAME))
        P[f"{side}_ankle"] += [_p(parts.foot(s), f"foot_{side}", ORANGE), _p(parts.foot_pad(s), f"foot_pad_{side}", PAD)]
    P["neck_pitch"] += [_p(parts.neck_link(), "neck_link", FRAME), _p(parts.neck_cover(), "neck_cover", YELLOW)]
    P["head_yaw"].append(_p(parts.pitch_bracket(), "pitch_bracket", FRAME))
    P["head_pitch"].append(_p(parts.roll_bracket(), "roll_bracket", FRAME))
    P["head_roll"] += [
        _p(parts.head_frame(), "head_frame", FRAME),
        _p(parts.head_shell(True), "head_shell_front", YELLOW), _p(parts.head_shell(False), "head_shell_back", YELLOW),
        _p(parts.eye_lcd(1), "eye_lcd_l", LCD), _p(parts.eye_lcd(-1), "eye_lcd_r", LCD),
        _p(parts.eye_lens(1), "eye_lens_l", GLASS), _p(parts.eye_lens(-1), "eye_lens_r", GLASS),
        _p(parts.camera(), "camera", ELEC), _p(parts.mics(), "mics", ELEC),
        _p(parts.beak_upper(), "beak_upper", ORANGE),
    ]
    P["beak"].append(_p(parts.beak_lower(), "beak_lower", ORANGE))
    return P


def build(pose=None, stages=None):
    """Assembly compound. `pose` = {joint: deg} rotates each stage about its axis (checks only)."""
    pose = pose or {}
    P = stages or stage_parts()

    def group(name):
        kids = list(P[name]) + [group(c) for c, (p, *_r) in JOINTS.items() if p == name]
        return bd.Compound(children=kids, label=f"{name}_stage")

    def posed(name, loc):
        o, a, _l = parts.joint_axis(name)
        rot = bd.Location((0, 0, 0), a, pose[name]) if name in pose else bd.Location()
        here = loc * bd.Pos(*o) * rot * bd.Pos(*(-v for v in o))
        kids = [here * p for p in P[name]] + [posed(c, here) for c, (p, *_r) in JOINTS.items() if p == name]
        return bd.Compound(children=kids, label=f"{name}_stage")

    roots = [n for n, (p, *_r) in JOINTS.items() if p is None]
    if pose:
        return bd.Compound(children=list(P[None]) + [posed(n, bd.Location()) for n in roots], label="robot_duck")
    return bd.Compound(children=list(P[None]) + [group(n) for n in roots], label="robot_duck")
