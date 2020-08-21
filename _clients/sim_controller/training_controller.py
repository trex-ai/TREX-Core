from _utils import utils

class TrainingController:
    """Training controller is a intended to provide very fine control of the trainin process, especially for generational training.
    Currently, it only controls which generation of weights is loaded.
    """
    def __init__(self, config, sim_status):
        self.__config = config
        self.__status = sim_status

    def load_curriculum(self, participant_id, current_generation):
        curriculum = {}
        # weights will always load on start or resume (running_generations == 0)
        # weights will only reload for non-learning agents beyond generation 0
        if self.__status['running_generations'] == 0 or \
                (self.__status['running_generations'] > 0 and participant_id not in self.__status['learning_agents']):
            curriculum['load_weights'] = True
        else:
            curriculum['load_weights'] = False

        if self.__config['study']['type'] != 'training':
            return curriculum

        gen = str(current_generation)
        curriculum.update(self.__config['training'][gen] if gen in self.__config['training'] else self.__config['training']['default'])

        if 'warm_up' in curriculum and curriculum['warm_up']:
            curriculum['gen_len'] = self.__config['study']['days'] * 1440
        return curriculum

    def select_generation(self, current_generation, selection_type=None):
        if selection_type == 'last':
            return current_generation - 1
        elif selection_type == 'random' and current_generation > 1:
            return utils.secure_random.randint(0, current_generation - 1)
        # selects last gen by default
        return current_generation - 1
