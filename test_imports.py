"""Test script to verify imports and syntax."""

try:
    from src.media_witch.cli.organize import organize_command
    from src.media_witch.features.organize.api import (OrganizeConfig,
                                                       organize_directory)
    from src.media_witch.ui.prompts import (ask_extras_classification,
                                            ask_nfo_overrides,
                                            ask_processing_choice, ask_season,
                                            ask_yes_no)
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
