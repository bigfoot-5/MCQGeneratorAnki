# Vendor Directory

This directory contains third-party Python packages bundled with this addon.

## Purpose

These packages are included to ensure the addon works regardless of what's
installed in the user's Python environment. This is especially important for
Anki addons since they run in Anki's Python environment, not the system Python.

## Regenerating

To update these dependencies, run:
    python3 bundle_dependencies.py

Or use Anki's Python for better compatibility:
    /Applications/Anki.app/Contents/MacOS/AnkiPython bundle_dependencies.py

## Shipping

This directory should be included when distributing the addon so users don't
need to install dependencies separately.
