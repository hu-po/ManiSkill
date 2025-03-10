import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["widowxai"])
class WidowXAIDraw(BaseAgent):
    uid = "widowxai_draw"
    urdf_path = f"{ASSET_DIR}/robots/widowxai/wxai_draw.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    ]
    ee_link_name = "pen_tip_link"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold