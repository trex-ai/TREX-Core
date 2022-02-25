import os
import argparse
if __name__ == '__main__':
    from _utils.runner.runner import Runner
    # configuration to be used must be under _configs
    # file name must be config name. Case sensitive.
    # it is sometimes possible to resume a simulation in case it crashes in the middle
    # however due to the complexity this is experimental and cannot be relied upon
    # TODO: Nov 29 2021; it may be nice to have the configs
    runner = Runner(config='envController_debugging', resume=False, purge=False)

    # list of simulations to be performed.
    # in general, it is recommended to perform at baseline and training at minimum
    simulations = [
        # {'simulation_type': 'baseline'},
        {'simulation_type': 'training'},
        # {'simulation_type': 'validation'}
    ]
    runner.run(simulations)