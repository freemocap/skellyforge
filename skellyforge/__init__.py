"""Top-level package for basic_template_repo."""

__package_name__ = "skellyforge"
__version__ = "v2024.08.1008"

__author__ = """Aaron Cherian"""
__email__ = "info@freemocap.org"
__repo_owner_github_user_name__ = "freemocap"
__repo_url__ = (
    f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
)
__repo_issues_url__ = f"{__repo_url__}issues"

import skellyforge.freemocap_utils as freemocap_utils
from skellyforge.freemocap_utils.config import default_settings
from skellyforge.freemocap_utils.constants import TASK_FILTERING, PARAM_CUTOFF_FREQUENCY, PARAM_SAMPLING_RATE, \
    PARAM_ORDER, PARAM_ROTATE_DATA, TASK_SKELETON_ROTATION, TASK_INTERPOLATION, TASK_FINDING_GOOD_FRAME

from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread


print(f"Thank you for using {__package_name__}!")
print(f"This is printing from: {__file__}")
print(f"Source code for this package is available at: {__repo_url__}")


