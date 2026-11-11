"""Compatibility entry point for the current final report; no provider calls."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('final_report.py')), run_name='__main__')
