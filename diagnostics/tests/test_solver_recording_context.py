"""Video alignment must be demonstrated, never assumed from filenames."""
import pytest
from scripts.solver_recording_context import check_video_times


def records():
    return [dict(number=i,time=10+i/6) for i in range(3)]


def test_video_times_use_recording_frame_numbers_and_relative_timestamps():
    check_video_times([0,1/6,2/6],records())


@pytest.mark.parametrize('times',[[0,1/6],[0,1/30,2/30],[1/6,2/6,3/6]])
def test_wrong_video_count_rate_or_offset_is_rejected(times):
    with pytest.raises(ValueError):check_video_times(times,records())


def test_missing_recording_frame_is_not_treated_as_next_video_frame():
    r=records();r[1]['number']=2
    with pytest.raises(ValueError):check_video_times([0,1/6,2/6],r)
