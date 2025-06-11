import os
from typing import Any, Dict
import dotenv


def str_to_bool(value: str) -> bool:
    """
    Converts a string to a boolean.
    Recognizes 'true', '1', 'yes', and 'y' as True;
    'false', '0', 'no', and 'n' as False.
    Raises ValueError if the input cannot be interpreted as a boolean.
    """
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    elif normalized in ("false", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Cannot convert {value} to bool")


class EnvSettings:
    _env_loaded = False  # Class-level flag to avoid reloading .env multiple times

    def __init__(self):
        if not EnvSettings._env_loaded:
            dotenv.load_dotenv()  # Loads .env if present; does nothing otherwise
            EnvSettings._env_loaded = True

        self.settings = self._get_settings_dict()

    def __repr__(self):
        return f"EnvSettings({self.settings})"

    @staticmethod
    def _normalize_value(value):
        """
        Converts common case-insensitive Boolean strings (e.g., "true", "False", "1", "0", "yes", "no")
        into Python booleans. Attempts to convert other numeric strings to integers,
        and leaves everything else as-is.
        """
        if not isinstance(value, str):
            return value

        lower_val = value.strip().lower()

        # Handle Boolean-like values
        if lower_val in ("true", "t", "1", "yes", "y", "on"):
            return True
        if lower_val in ("false", "f", "0", "no", "n", "off"):
            return False

        # Handle integer values
        try:
            return int(value)
        except ValueError:
            # Return the original string if not convertible to int
            return value

    def _get_settings_dict(self) -> Dict[str, Any]:
        """
        Loads all environment settings from the .env file and os.environ without knowing the key names.
        This method first reads values from .env using dotenv.dotenv_values() and then overlays
        any values found in os.environ (which have higher precedence).
        All values are normalized (e.g., booleans converted to True/False, numeric strings to int).
        """
        # Load all key-value pairs from the .env file (if present)
        dot_env_values = dotenv.dotenv_values()  # returns a dict from the .env file
        # Merge with os.environ so that system environment variables override the .env file values.
        combined = {**dot_env_values, **dict(os.environ)}
        normalized = {}
        for key, value in combined.items():
            normalized[key] = self._normalize_value(value)
        return normalized

    def get_settings_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary of settings.
        """
        return self.settings

    def get_setting(self, setting: str, default: Any = None) -> Any:
        """
        Returns the setting value for the given key.
        Checks os.environ live before falling back to the internal settings dict and default.
        """
        if setting in os.environ:
            return self._normalize_value(os.environ[setting])
        return self.settings.get(setting, default)

    # Alias for get_setting
    def get(self, setting: str, default: Any = None) -> Any:
        return self.get_setting(setting, default)

    def is_setting_on(self, setting: str) -> bool:
        """
        Returns True if the setting is enabled (True), False otherwise.
        """
        return self.get_setting(setting) is True

    def is_configured(self, setting: str) -> bool:
        """
        Returns True if the environment setting exists.
        """
        return self.get_setting(setting) is not None

    def override_setting(self, setting: str, value: Any):
        """
        Overrides the value of a setting both in the current cache
        and in the real environment, so that all new EnvSettings
        instances pick it up immediately.
        """
        # Update the in-memory settings dict
        self.settings[setting] = value

        # Export to the OS environment for subsequent instances
        import os

        os.environ[setting] = str(value)
