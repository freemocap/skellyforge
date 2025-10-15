from pathlib import Path


def get_videos_from_folder(video_folder: str | Path) -> list[Path]:
    """Search the folder for 'mp4' files (case insensitive) and return them as a list"""
    list_of_video_paths = list(Path(video_folder).glob("*.mp4")) + list(
        Path(video_folder).glob("*.MP4")
    )
    unique_list_of_video_paths = get_unique_list(list_of_video_paths)

    unique_list_of_video_paths.sort()

    return unique_list_of_video_paths


def get_unique_list(list: list) -> list:
    """Return a list of the unique elements from input list"""
    unique_list = []
    [unique_list.append(element) for element in list if element not in unique_list]

    return unique_list