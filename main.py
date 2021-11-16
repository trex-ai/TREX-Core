import os

if __name__ == '__main__':
    from _utils.runner.runner import Runner
    # configuration to be used must be under _configs
    # file name must be config name. Case sensitive.
    # it is sometimes possible to resume a simulation in case it crashes in the middle
    # however due to the complexity this is experimental and cannot be relied upon
    runner = Runner(config='config_name', resume=False, purge=False)

    # list of simulations to be performed.
    # in general, it is recommended to perform at baseline and training at minimum
    simulations = [
        {'simulation_type': 'baseline'},
        # {'simulation_type': 'training'},
        # {'simulation_type': 'validation'}
    ]
    runner.run(simulations)