from typing import List
from fsm.fsm_state import FSMState

from .passive_mode import PassiveMode
from .fixed_pose import FixedPose
from .loco_mode import LocoMode
from .loco_mode_pt import LocoModePt
# from .jntm_motion import JntmMotion
# from .unistore_motion import UsMotion

mode_list: List[FSMState] = [
    PassiveMode,
    FixedPose,
    LocoMode,
    LocoModePt,
    # JntmMotion,
    # UsMotion,
]