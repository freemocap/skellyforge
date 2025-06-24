from enum import Enum
from skellymodels.managers.actor import Actor
from skellymodels.models.tracking_model_info import ModelInfo
from skellymodels.models.aspect import Aspect
import numpy as np

##NOTE: Playing around with the idea of making a Board Actor. Untested, but I think this could be extended to 
## keep track of 7x5 and 5x3 data (or take in those parameters) when saving out to automatically create the correct Board
## will return to this later

class BoardAspectEnum(Enum):
    BOARD = "board"

class Board(Actor):
    def __init__(self, name:str, model_info:ModelInfo):
        super().__init__(name, model_info)
        self._add_board()
    
    def _add_board(self):
        self.aspect_from_model_info(
            name = BoardAspectEnum.BOARD.value
        )

    @property
    def board(self) -> Aspect:
        return self.aspects[BoardAspectEnum.BOARD.value]
    
    def add_tracked_points_numpy(self, tracked_points_numpy_array):
        self.board.add_tracked_points(
            tracked_points_numpy_array[:, self.tracked_point_slices[BoardAspectEnum.BOARD.value],:]
        )

    @classmethod
    def from_board_dimensions(rows:int, columns:int):
        """
        potential example of how we could build a board from the input dimensions (will return to this later)
        """
        pass