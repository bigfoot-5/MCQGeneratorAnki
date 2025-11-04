#!/usr/bin/env python3
"""
Bundles dependencies for Anki addon into vendor/ directory.

This script installs the required packages into a vendor/ directory
that will be shipped with the addon, ensuring all dependencies are
self-contained.

Usage:
    python3 bundle_dependencies.py
    
    # Or with Anki's Python (recommended for compatibility):
    /Applications/Anki.app/Contents/MacOS/AnkiPython bundle_dependencies.py
"""
import os
import sys
import subprocess
import shutil

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ADDON_DIR, 'vendor')
REQUIREMENTS_FILE = os.path.join(ADDON_DIR, 'requirements.txt')

def clean_vendor_dir():
    """Remove existing vendor directory"""
    if os.path.exists(VENDOR_DIR):
        print(f"Removing existing {VENDOR_DIR}...")
        shutil.rmtree(VENDOR_DIR)
    os.makedirs(VENDOR_DIR)
    print(f"Created {VENDOR_DIR}")

def install_dependencies():
    """Install dependencies to vendor directory"""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"ERROR: {REQUIREMENTS_FILE} not found!")
        sys.exit(1)
    
    print(f"\nInstalling dependencies from {REQUIREMENTS_FILE}...")
    print(f"Using Python: {sys.executable}\n")
    
    # Install packages to vendor directory
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            '--target', VENDOR_DIR,
            '--upgrade',
            '-r', REQUIREMENTS_FILE
        ])
        print("\n✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        sys.exit(1)

def cleanup():
    """Remove unnecessary files from vendor directory"""
    print("\nCleaning up vendor directory...")
    
    for root, dirs, files in os.walk(VENDOR_DIR):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path)
            dirs.remove('__pycache__')
        
        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith(('.pyc', '.pyo')):
                os.remove(os.path.join(root, file))
        
        # Remove .dist-info and .egg-info directories
        dirs_to_remove = [d for d in dirs if d.endswith(('.dist-info', '.egg-info'))]
        for dir_name in dirs_to_remove:
            shutil.rmtree(os.path.join(root, dir_name))
            dirs.remove(dir_name)
    
    print("✓ Cleanup complete")

def create_vendor_readme():
    """Create a README in vendor directory explaining its purpose"""
    readme_content = """# Vendor Directory

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
"""
    readme_path = os.path.join(VENDOR_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print("✓ Created vendor/README.md")

def main():
    print("=" * 60)
    print("Bundling dependencies for Anki addon")
    print("=" * 60)
    print(f"Addon directory: {ADDON_DIR}")
    print(f"Vendor directory: {VENDOR_DIR}")
    
    clean_vendor_dir()
    install_dependencies()
    cleanup()
    create_vendor_readme()
    
    print("\n" + "=" * 60)
    print("✓ Bundling complete!")
    print("=" * 60)
    print(f"\nDependencies are now in: {VENDOR_DIR}")
    print("\nImportant:")
    print("  - Include the vendor/ directory when shipping your addon")
    print("  - Users don't need to install anything separately")
    print("  - The addon will automatically use vendored packages")

if __name__ == '__main__':
    main()
