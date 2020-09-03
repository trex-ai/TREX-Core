import os

if __name__ == '__main__':
    from _utils.sim_maker import Maker
    # configuration to be used must be under _configs
    # file name must be config name. Case sensitive.
    # it is sometimes possible to resume a simulation in case it crashes in the middle
    # however due to the complexity this is experimental and cannot be relied upon
    maker = Maker(config='test_if_baselines', resume=False)

    # list of simulations to be performed.
    # in general, it is recommended to perform at baseline and training at minimum
    simulations = [
        {'type': 'baseline'},
        {'type': 'training'}
        # {'type': 'validation'}
    ]
    maker.launch(simulations)
