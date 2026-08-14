# Frozen legacy copy (2026-08-14). These definitions once shadowed
# ``skellytracker.trackers.base_tracker.base_tracker_abcs`` behind a silent
# try/except import. That module no longer exists in skellytracker (deleted in
# the mapping rework), so the import could never succeed and every import
# silently fell back — the silent-fallback mechanism is removed, and these are
# now the only definitions. The posthoc path (``data_models``) still consumes
# this module until the posthoc rebuild deletes the old architecture (Phase E).
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

TrackedPointIdString = str
TrackerTypeString = str
TrackedPoint2d = np.ndarray


class BaseObservation(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_number: (
        int  # the frame number of the image in which this observation was made
    )
    tracker_type: TrackerTypeString = Field(
        description="Name of the tracker that made this observation."
    )

    @classmethod
    @abstractmethod
    def from_detection_results(cls, *args, **kwargs):
        pass

    @abstractmethod
    def to_tracked_points(
        cls, *args, **kwargs
    ) -> dict[TrackedPointIdString, TrackedPoint2d]:
        pass

    @abstractmethod
    def to_array(self) -> np.ndarray:
        pass

    def to_json_string(self) -> str:
        return json.dumps(self.model_dump_json(), indent=4)

    def to_json_bytes(self) -> bytes:
        return self.to_json_string().encode("utf-8")


BaseObservations = list[BaseObservation]


class BaseRecorder(BaseModel, ABC):
    # TODO: could be called ObservationGroup
    observations: List[BaseObservation] = Field(default_factory=list)

    def add_observation(self, observation: BaseObservation):
        self.observations.append(observation)

    def add_observations(self, observations: List[BaseObservation]):
        self.observations.extend(observations)

    # I'm imagining these can be used if you want the data but want to handle saving elsewhere
    @property
    def to_array(self) -> np.ndarray:
        return np.stack([observation.to_array() for observation in self.observations])

    @property
    def to_json_string(self) -> str:
        output_dict = {
            frame_number: observation.model_dump_json()
            for frame_number, observation in enumerate(self.observations)
        }
        return json.dumps(output_dict, indent=4)

    # and these are used if you want skellytracker to handle the saving
    def save_array(self, output_path: Path):
        np.save(file=output_path, arr=self.to_array)

    def save_json_file(self, output_path: Path):
        with open(output_path, "w") as json_file:
            json_file.write(self.to_json_string)

    def clear(self):
        self.observations = []
