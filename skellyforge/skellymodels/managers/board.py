from enum import Enum
from skellymodels.managers.actor import Actor
from skellymodels.models.tracking_model_info import ModelInfo
from skellymodels.models.aspect import Aspect
from pathlib import Path
##NOTE: Playing around with the idea of making a Board Actor. Untested, but I think this could be extended to 
## keep track of 7x5 and 5x3 data (or take in those parameters) when saving out to automatically create the correct Board
## will return to this later

class BoardAspectEnum(Enum):
    BODY = "body"

class Board(Actor):
    def __init__(self, name:str, model_info:ModelInfo):
        super().__init__(name, model_info)
        self._add_board()
    
    def _add_board(self):
        self.aspect_from_model_info(
            name = BoardAspectEnum.BODY.value
        )

    @property
    def body(self) -> Aspect:
        return self.aspects[BoardAspectEnum.BODY.value]
    
    def add_tracked_points_numpy(self, tracked_points_numpy_array):
        self.body.add_tracked_points(
            tracked_points_numpy_array[:, self.tracked_point_slices[BoardAspectEnum.BODY.value],:]
        )

    @classmethod
    def from_board_definition(cls, columns: int, rows:int):
        board_yaml = f"charuco_board_{columns}_{rows}.yaml"
        try:
            path_to_yaml = Path(__file__).parents[1]/'tracker_info'/board_yaml
            charuco_model_info = ModelInfo.from_config_path(path_to_yaml)
            board = cls(
                name = f"charuco_board_{columns}_{rows}",
                model_info = charuco_model_info
            )
            return board
        except FileNotFoundError:
            print("Charuco board definition not recognized")
