"""Top-level package for basic_template_repo."""

__package_name__ = "skellyforge"
__version__ = "v2024.12.1009"

__author__ = """Aaron Cherian"""
__email__ = "info@freemocap.org"
__repo_owner_github_user_name__ = "freemocap"
__repo_url__ = (
    f"https://github.com/{__repo_owner_github_user_name__}/{__package_name__}/"
)
__repo_issues_url__ = f"{__repo_url__}issues"

from skellyforge.system.logging_configuration.log_levels import LogLevels

LOG_LEVEL = LogLevels.TRACE
