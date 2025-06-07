import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.robots.tatbot.tatbot import Tatbot

robot = Tatbot(TatbotConfig())

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id="user/tatbot" + str(int(time.time())),
    fps=10,
    features=dataset_features,
    robot_type=robot.name,
)

robot.connect()

MAX_STEPS = 100
i = 0
while i < MAX_STEPS:

    action_sent = robot.send_action({
        "left.joint_0.pos": 0.0,
        "left.joint_1.pos": 0.0,
        "left.joint_2.pos": 0.0,
        "left.joint_3.pos": 0.0,
        "left.joint_4.pos": 0.0,
        "left.joint_5.pos": 0.0,
        "left.gripper.pos": 0.0,
        "right.joint_0.pos": 0.0,
        "right.joint_1.pos": 0.0,
        "right.joint_2.pos": 0.0,
        "right.joint_3.pos": 0.0,
        "right.joint_4.pos": 0.0,
        "right.joint_5.pos": 0.0,
        "right.gripper.pos": 0.0,
    })
    observation = robot.get_observation()

    frame = {**action_sent, **observation}
    task = "Sleeping"

    dataset.add_frame(frame, task)
    i += 1

robot.disconnect()
dataset.save_episode()
dataset.push_to_hub()
