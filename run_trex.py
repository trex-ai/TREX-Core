'''
This is the first test, can I run a sim from another file in the same directory with the same interpreter
'''
from pprint import pprint
# # import subprocess
# # testProcess = subprocess.run(['C:/source/TREX_Core/venv/Scripts/python.exe', 'main.py'], timeout=10)
# from _utils.runner.runner import Runner
# runner = Runner(config='TB8', resume=False, purge=False)
#
# simulations = [{'simulation_type': 'training'}]
# launchlist = runner.run(simulations, run=False)
#
# pprint(launchlist)
from setuptools import find_packages

pprint(find_packages())
from _utils.runner.runner import Runner
run = Runner()