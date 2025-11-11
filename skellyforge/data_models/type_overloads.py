from skellyforge.data_models.trajectory_2d import Trajectory2d
from skellyforge.data_models.observation import BaseObservation

# TODO: change to base models, add validation during class methods
CameraIdString = str
FrameNumber = int
FrameGroup = dict[CameraIdString, BaseObservation]
FrameGroups = dict[FrameNumber, FrameGroup]

Trajectory2dGroup = dict[CameraIdString, Trajectory2d]