from enum import Enum
import numpy as np
from skellymodels.models.aspect import Aspect
from skellymodels.managers.animal import Animal
from skellymodels.models.tracking_model_info import ModelInfo
class HumanAspectNames(Enum):
    BODY = "body"
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"

HUMAN_ENUMS = (HumanAspectNames.FACE,
                HumanAspectNames.LEFT_HAND,
                HumanAspectNames.RIGHT_HAND)

class Human(Animal):
    """
    Specialized Actor class representing a tracked human subject.

    In addition to the 'body' aspect defined in the 'Animal' base class,
    this class adds support for 'face', 'left_hand', and 'right_hand' aspects,
    depending on the configuration in the provided 'ModelInfo'

    Parameters
    ----------
    name : str
        Identifier for the human actor (e.g., subject ID, session name).
    model_info : ModelInfo
        Configuration describing marker layout and aspect slicing
        from the full tracker output.

    Attributes
    ----------
    face : Aspect or None
        Face aspect if present in the model config.
    left_hand : Aspect or None
        Left hand aspect if present in the model config.
    right_hand : Aspect or None
        Right hand aspect if present in the model config.        
    """
    def __init__(self, name: str, model_info:ModelInfo):
        super().__init__(name, model_info)
    
        self._initialize_aspects()

    def _initialize_aspects(self):
        """
        Initializes the anatomical aspects defined in `HUMAN_ENUMS`,
        adding them if they exist in the current `aspect_order`.
        """

        for aspect_enum in HUMAN_ENUMS:
            if aspect_enum.value in self.aspect_order:
                self.aspect_from_model_info(
                    name = aspect_enum.value
                )
    
    @property
    def face(self) -> Aspect|None:
        """
        Returns the face aspect, if available.

        Returns
        -------
        Aspect or None
        """
        return self.aspects.get(HumanAspectNames.FACE.value)
    
    @property
    def left_hand(self) -> Aspect|None:
        """
        Returns the left hand aspect, if available.

        Returns
        -------
        Aspect or None
        """
        return self.aspects.get(HumanAspectNames.LEFT_HAND.value)
    
    @property
    def right_hand(self) -> Aspect|None:
        """
        Returns the right hand aspect, if available.

        Returns
        -------
        Aspect or None
        """
        return self.aspects.get(HumanAspectNames.RIGHT_HAND.value)
    
    def add_tracked_points_numpy(self, tracked_points_numpy_array:np.ndarray):
        """
        Splits the full tracked point array into aspects and adds each
        subset to its respective Aspect object.

        The `body` data is handled by the superclass (`Animal`), and
        this method additionally processes face and hand aspects if
        defined in the model.

        Parameters
        ----------
        tracked_points_numpy_array : ndarray of shape (F, M, 3)
            Full 3D marker array from the tracker.
        """

        super().add_tracked_points_numpy(tracked_points_numpy_array)

        for aspect_enum in HUMAN_ENUMS:
            
            aspect = self.aspects.get(aspect_enum.value)
            if aspect_enum.value in self.tracked_point_slices and aspect:
                aspect.add_tracked_points(
                    tracked_points_numpy_array[:, self.tracked_point_slices[aspect_enum.value],:]
                )

    def add_reprojection_error_numpy(self, reprojection_error_data: np.ndarray):
        """
        Adds per-marker reprojection error data to each aspect, if available.

        The `body` errors are handled by the superclass (`Animal`), and
        this method additionally processes face and hand aspects.
        
        Parameters
        ----------
        reprojection_error_data : ndarray of shape (F, M)
            2D array of reprojectio
        """
        super().add_reprojection_error_numpy(reprojection_error_data)

        for aspect_enum in HUMAN_ENUMS:
            aspect = self.aspects.get(aspect_enum.value)
            if aspect_enum.value in self.tracked_point_slices and aspect:
                aspect.add_reprojection_error(
                    reprojection_error_data[:, self.tracked_point_slices[aspect_enum.value]]
                )
