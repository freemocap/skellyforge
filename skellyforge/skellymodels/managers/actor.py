from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import datetime
from skellymodels.models.aspect import Aspect
from skellymodels.models.trajectory import Trajectory
from typing import Dict, Optional

from skellymodels.models.tracking_model_info import ModelInfo
from skellymodels.biomechanics.anatomical_calculations import CalculationPipeline, STANDARD_PIPELINE

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FREEMOCAP_PARQUET_NAME = 'freemocap_data_by_frame.parquet'
class Actor(ABC):
    """
    The Actor class is a container for multiple *aspects* (e.g. body, face, hands) that belong to a single actor,
    which is the thing being tracked in 3D (human, animal, etc.)
    
    Parameters
    ----------
    name : str
        Identifier for this actor
    model_info: ModelInfo
        Parsed YAML configuration describing tracker ouput and anatomy of the actor
    
    Attributes
    ----------
    name: str
        Actor identifier
    aspects: dict[str, Aspects]
        Maps aspect name to an Aspect instance
    tracker: str
        Name of the underlying pose estimation model (e.g. 'mediapipe'), taken from 
        the model_info
    aspect_order: list[str]
        The aspect order used when slicing the raw tracker-output data (e.g. specifiying
        whether the raw data comes in body/hands/face order or body/face/hands and so on)
    tracker_point_slices: dict[str, slice]
        Slices into the raw tracked points array for each aspect
    model_info: ModelInfo
        Original configuration object
    """

    def __init__(self, name: str, model_info: ModelInfo):
        self.name = name
        self.aspects: Dict[str, Aspect] = {}
        self.tracker = model_info.name #the day we allow for separate pose estimation for different things (body/hands/face), we (I) may regret this
        self.aspect_order = model_info.order
        self.tracked_point_slices = model_info.tracked_point_slices
        self.model_info = model_info

    def __getitem__(self, key: str):
        """
        Use to access aspects directly (i.e. actor['body'] instead of actor.aspect['body'])
        """
        return self.aspects[key]

    def __str__(self) -> str:
        lines = [f"Actor with name:{self.name!r} and tracker:{self.model_info.name!r}"]

        for asp_name, asp in self.aspects.items():
            traj_names = ", ".join(t.name for t in asp.trajectories.values()) or "∅"
            lines.append(f"  • {asp_name:<10}  trajectories: {traj_names}")

        if not self.aspects:
            lines.append("  (no aspects)")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def add_tracked_points_numpy(self, tracked_points_numpy_array: np.ndarray):
        """
        Ingest a ``(num_frames, num_markers, 3)`` array of 3D joint position data and distributes the data to
        the correct Aspect istances of the class

        Parameters
        ----------
        tracked_points_numpy_array : ndarray, shape (F, M, 3)
            Raw 3-D tracker output in the order defined by the YAML
            file.
        """
        pass
    
    @classmethod
    def from_tracked_points_numpy_array(cls, name: str, model_info: ModelInfo, tracked_points_numpy_array: np.ndarray) -> "Actor":
        """
        Convenience wrapper to instantiate an ``Actor`` and feed it tracker data
        """
        actor = cls(name=name, model_info=model_info)
        actor.add_tracked_points_numpy(tracked_points_numpy_array=tracked_points_numpy_array)
        return actor
    
    @classmethod
    def from_data(cls, path_to_data_folder: Path|str, model_info: ModelInfo|None = None) -> "Actor":
        """
        Convenience wrapper to instantiate an ``Actor`` from previously saved out output data

        The function currently expects a file named ``freemocap_data_by_frame.parquet`` inside
        *path_to_data_folder*.
        """
        #Later on, if needed, we can consider fallback methods to loading from the big CSV or all the individual CSVs as well
        try:
            parquet_file = path_to_data_folder / FREEMOCAP_PARQUET_NAME
            return cls.from_parquet(path_to_parquet_file = parquet_file)
        except FileNotFoundError:
            logger.warning(f"Could not find parquet file at {parquet_file}")
        except Exception as e:
            logger.warning(f"Failed to load from Parquet: {e}")
            
        raise RuntimeError(f"Could not load data from {path_to_data_folder}")
    
    @classmethod
    def from_parquet(cls, path_to_parquet_file: Path|str, model_info: ModelInfo|None = None) -> "Actor":
        """
        Convience wrapper to instantiate an ``Actor`` from the ``freemocap_data_by_frame.parquet`` file
        """
        path_to_parquet_file = Path(path_to_parquet_file)
        dataframe = pd.read_parquet(path_to_parquet_file)
        if not model_info:
            if 'model_info' not in dataframe.attrs:
                raise ValueError("No model_info found in parquet file, please provide a ModelInfo instance")
            model_info = ModelInfo.from_model_dict(dataframe.attrs['model_info'])
            
        actor = cls(name =dataframe.attrs['model_info']['name'],
                    model_info = model_info)

        if set(actor.aspect_order) != set(dataframe.attrs['model_info']['order']): #Want to come back around and make a more robust check
            raise ValueError(f"Aspects in parquet file {dataframe.attrs['model_info']['order']} do not match aspects specified in model info {actor.aspect_order}")

        actor.populate_aspects_from_parquet(dataframe)
        return actor
    
    def add_aspect(self, aspect: Aspect):
        """
        Add an Aspect instance to the actor 
        """
        self.aspects[aspect.name] = aspect
        
    def aspect_from_model_info(self, name:str) -> None:
        """
        Creates a structured Aspect from the model_info configuration. This Aspect will
        have a defined/validated AnatomicalStructure (marker names, virtual markers, center of mass, etc.)
        that will be used for data
        """
        aspect:Aspect = Aspect.from_model_info(
            name = name,
            model_info = self.model_info,
            metadata = {"tracker_type": self.tracker}
        )
        self.add_aspect(aspect)

    def calculate(self, pipeline:CalculationPipeline = STANDARD_PIPELINE):
        """
        Runs the biomechanics pipeline to for anatomical calculations (i.e. center of mass calculation, rigid bones enforcement)
        """
        
        for aspect in self.aspects.values():
            results_logs = pipeline.run(aspect=aspect)

            logger.info(f"\nResults for aspect {aspect.name}:")
            for msg in results_logs:
                logger.info(f"  {msg}")

    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Collect every trajectory from every aspect into one tidy-formatted DataFrame
        """
        all_data = []

        # Loop through aspects
        for aspect_name, aspect in self.aspects.items():
            for trajectory_name, trajectory in aspect.trajectories.items():

                # Convert trajectory to a DataFrame
                trajectory_df = trajectory.as_dataframe
                
                # Add metadata columns
                trajectory_df['model'] = f"{aspect.metadata['tracker_type']}.{aspect_name}"
                trajectory_df['trajectory'] = trajectory_name  # Store the trajectory type

                # Add error column
                if aspect.reprojection_error is None:
                    trajectory_df['reprojection_error'] = np.nan
                else:
                    trajectory_df['reprojection_error'] = trajectory_df.apply(
                        lambda row: aspect.reprojection_error.get_frame(frame_number=row['frame']).get(row['keypoint'], np.nan),
                        axis=1
                    )  

                # Append DataFrame to the list
                all_data.append(trajectory_df)

        # Combine all DataFrames into one
        big_df = pd.concat(all_data, ignore_index=True)

        # Sort by frame, model, and type
        big_df = big_df.sort_values(by=['frame', 'model', 'trajectory']).reset_index(drop=True)

        return big_df
    
    def create_summary_dataframe_with_metadata(self) -> pd.DataFrame:
        """
        Add metadata to the summary dataframe for saving out a Parquet
        """
        df = self.create_summary_dataframe()
        df.attrs['metadata'] = {
            'created_at': datetime.datetime.now().isoformat(),
            'created_with': 'skelly_models',
        }

        df.attrs['model_info'] = self.model_info.model_dump()
        return df

    def _set_output_folder(self, path_to_output_folder: Path|str|None = None) -> Path:
        path_to_output_folder = Path.cwd() if path_to_output_folder is None else Path(path_to_output_folder)
        return path_to_output_folder

    def save_out_numpy_data(self, path_to_output_folder: Path|str|None = None):
        """
        Saves out a .npy file for each Trajectory in each Aspect with format {tracker_type}_{aspect}_{trajectory} 
        (i.e. 'mediapipe_body_3d_xyz')
        """
        path_to_output_folder = self._set_output_folder(path_to_output_folder)

        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                save_path = path_to_output_folder / f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.npy"
                np.save(save_path,
                        trajectory.as_array) 
                logger.info(f"Saved out {save_path}")

    def save_out_csv_data(self, path_to_output_folder: Path|str|None = None):
        """
        Saves out a .csv file for each Trajectory in each Aspect with format {tracker_type}_{aspect}_{trajectory} 
        (i.e. 'mediapipe_body_3d_xyz')
        """
        path_to_output_folder = self._set_output_folder(path_to_output_folder)

        for aspect in self.aspects.values():
            for trajectory in aspect.trajectories.values():
                save_path = path_to_output_folder / f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv"
                trajectory.as_dataframe.to_csv(path_to_output_folder/f"{aspect.metadata['tracker_type']}_{aspect.name}_{trajectory.name}.csv", index = False)
                logger.info(f"Saved out {save_path}") 

    def save_out_all_data_csv(self, path_to_output_folder: Path|str|None = None):
        """
        Saves out a CSV in tidy format with all Trajectories from all Aspects
        """
        path_to_output_folder = self._set_output_folder(path_to_output_folder)

        save_path = path_to_output_folder / 'freemocap_data_by_frame.csv'    
        self.create_summary_dataframe().to_csv(save_path, index=False)
        logger.info(f"CSV successfully saved to {save_path}")

    def save_out_all_data_parquet(self, path_to_output_folder: Path|str|None = None):
        """
        Saves out a Parquet file using the same dataframe created in `create_summary_dataframe` and 
        adds additional metadata to the file.
        """
        path_to_output_folder = self._set_output_folder(path_to_output_folder)

        dataframe = self.create_summary_dataframe_with_metadata()
        save_path = path_to_output_folder / FREEMOCAP_PARQUET_NAME
        dataframe.to_parquet(save_path)
        logger.info(f"Parquet successfully saved to {save_path}")

    def save_out_all_xyz_numpy_data(self, path_to_output_folder: Path|str|None = None):
        """
        Saves out a single .npy file with all xyz trajectories from all aspects
        """
        path_to_output_folder = self._set_output_folder(path_to_output_folder)

        all_xyz_data = np.concatenate([self.aspects[aspect_name].xyz.as_array for aspect_name in self.aspect_order], axis = 1)

        save_path = path_to_output_folder/f"{self.tracker}_skeleton_3d.npy"
        np.save(save_path, all_xyz_data)
        logger.info(f"Combined marker position numpy array saved to f{save_path}")
        f = 2

    def populate_aspects_from_parquet(self, dataframe:pd.DataFrame):
        """
        Used when running the `from_parquet` class method to sort through the Parquet and distribute the data
        back into Aspects and Trajectories that can be added to the actor 
        """
        num_frames = dataframe['frame'].nunique()
        expected_models = {f"{self.tracker}.{aspect_name}" for aspect_name in self.aspects}

        for model_name, aspect_data in dataframe.groupby('model'): #model name is formatted {tracker}.{aspect_name} in our CSV/parquet
            if model_name not in expected_models:
                raise ValueError(f"Aspect {model_name} not found in aspects initialized in Actor: {expected_models}")
            
            tracker_name, aspect_name = model_name.split(".")
            trajectory_dict: dict[str, Trajectory] = {}

            for trajectory_name, trajectory_data in aspect_data.groupby('trajectory'):
                marker_order = trajectory_data["keypoint"].drop_duplicates().tolist()

                num_markers = len(marker_order)

                trajectory_data_wide = (
                    trajectory_data
                    .pivot_table(index="frame", columns="keypoint", values=["x", "y", "z"], dropna = False)
                    .swaplevel(axis=1)
                    .sort_index(axis=1)
                    .reindex(columns = marker_order, level = 0)
                    )
                
                trajectory_array = trajectory_data_wide.to_numpy().reshape(num_frames,num_markers,3)
                
                trajectory = Trajectory(
                    name = trajectory_name,
                    array = trajectory_array,
                    landmark_names = marker_order
                )

                trajectory_dict[trajectory_name] = trajectory
                
            self.aspects.get(aspect_name).add_trajectory(trajectory_dict)