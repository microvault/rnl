from typing import List

class ConfigRobot():
    def __init__(self, radius, backwards, typeRobot, vel_linear, val_angular):

      self.radius = radius
      self.backwards = backwards
      self.typeRobot = typeRobot
      self.vel_linear = vel_linear
      self.val_angular = val_angular

    def param(self) -> List[float]:
      return [self.radius, self.backwards, self.typeRobot, self.vel_linear, self.val_angular]

