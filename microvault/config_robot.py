from typing import Optional

import sentry_sdk

sentry_sdk.init(
    dsn="https://a543aaf28e1fde91541d9ee60cb16951@o4506720636502016.ingest.sentry.io/4506720639057920",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

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
