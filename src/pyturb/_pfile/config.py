"""
Configuration parsing for ODAS setup strings.
"""

import re
from typing import Dict, Optional


class SetupConfig:
    """Parser for ODAS INI-style configuration strings with duplicate section support."""

    def __init__(self, config_string: str):
        self.config_string = config_string
        self._parse_config()

    def _parse_config(self):
        """Parse the INI-style configuration string with support for duplicate sections."""
        # ODAS config files can have duplicate section names (e.g., multiple [channel] sections)
        # Store as a list of dicts: [{'section': str, 'params': {key: value}}]
        self.sections = []

        current_section = "root"
        current_params = {}

        lines = self.config_string.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(";"):
                continue

            # Remove inline comments
            if ";" in line:
                line = line.split(";")[0].strip()

            # Check for section header
            if line.startswith("[") and line.endswith("]"):
                # Save previous section if it has content
                if current_params or current_section == "root":
                    self.sections.append(
                        {"section": current_section.lower(), "params": current_params}
                    )

                # Start new section
                current_section = line[1:-1].strip()
                current_params = {}
            else:
                # Parse parameter = value
                if "=" in line:
                    key, value = line.split("=", 1)
                    current_params[key.strip().lower()] = value.strip()

        # Don't forget the last section
        if current_params or current_section != "root":
            self.sections.append(
                {"section": current_section.lower(), "params": current_params}
            )

    def get_sections(self, pattern: str = "") -> list:
        """
        Get section names matching a pattern.

        Parameters
        ----------
        pattern : str
            Regex pattern to match section names. Empty string returns all sections.

        Returns
        -------
        list
            List of matching section names (may include duplicates)
        """
        section_names = [s["section"] for s in self.sections]
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            section_names = [s for s in section_names if regex.search(s)]
        return section_names

    def get_section_dicts(self, section_name: str) -> list:
        """
        Get all section dictionaries matching a section name.

        Parameters
        ----------
        section_name : str
            Section name to search for

        Returns
        -------
        list
            List of matching section dictionaries with their parameters
        """
        section_name = section_name.lower()
        return [s for s in self.sections if s["section"] == section_name]

    def get_value(
        self, section: str, parameter: str, default=None, index: int = 0
    ) -> Optional[str]:
        """
        Get a parameter value from a section.

        Parameters
        ----------
        section : str
            Section name
        parameter : str
            Parameter name
        default : optional
            Default value if not found
        index : int
            Index of the section if there are duplicates (default: 0 for first match)

        Returns
        -------
        str or None
            Parameter value or default
        """
        section = section.lower()
        parameter = parameter.lower()

        # Find matching sections
        matching = [s for s in self.sections if s["section"] == section]

        if not matching or index >= len(matching):
            return default

        return matching[index]["params"].get(parameter, default)

    def get_channel_params(self, channel_name: str) -> Optional[Dict]:
        """
        Get all parameters for a channel by name.

        Searches through all channel sections to find one matching the given name.

        Parameters
        ----------
        channel_name : str
            Channel name to search for (e.g., 'T1_dT1', 'sh1', 'P')

        Returns
        -------
        dict or None
            Dictionary of channel parameters, or None if not found
        """
        channel_name_lower = channel_name.lower()

        for section in self.sections:
            if section["section"] == "channel":
                params = section["params"]
                # Check if 'name' matches
                name = params.get("name", "").lower()
                if name == channel_name_lower:
                    return params

        return None
