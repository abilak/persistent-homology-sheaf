"""Kept for backward compatibility: python experiments_rebuttal/compare_zinc.py [baseline_variant]"""
import os, sys, runpy
sys.argv = [sys.argv[0], "ZINC"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_fixed.py"), run_name="__main__")
