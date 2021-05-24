import os

if __name__ == '__main__':
    from _utils.runner.runner import Runner
    # configuration to be used must be under _configs
    # file name must be config name. Case sensitive.
    # it is sometimes possible to resume a simulation in case it crashes in the middle
    # however due to the complexity this is experimental and cannot be relied upon
<<<<<<< HEAD
    runner = Runner(config='remote_agent_test', resume=False)
=======
    runner = Runner(config='config_name', resume=False, purge=False)
>>>>>>> master

    # list of simulations to be performed.
    # in general, it is recommended to perform at baseline and training at minimum
    simulations = [
<<<<<<< HEAD
        # {'simulation_type': 'baseline'},
        {'simulation_type': 'training'}
        # {'type': 'validation'}
=======
        {'simulation_type': 'baseline'},
        # {'simulation_type': 'training'},
        # {'simulation_type': 'validation'}
>>>>>>> master
    ]
    runner.run(simulations)