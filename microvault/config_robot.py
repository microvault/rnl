from typing import Optional

import sentry_sdk


class ConfigRobot:
    def __init__(
        self,
        *,
        radius: Optional[float] = None,
        backwards: Optional[bool] = None,
        typeRobot: Optional[str] = None,
        vel_linear: Optional[float] = None,
        val_angular: Optional[float] = None
    ):
        self.radius = radius
        self.backwards = backwards
        self.typeRobot = typeRobot
        self.vel_linear = vel_linear
        self.val_angular = val_angular
