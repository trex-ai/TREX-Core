from _utils import utils

class TrainingController:
    """Training controller is a intended to provide very fine control of the trainin process, especially for generational training.
    Currently, it only controls which generation of weights is loaded.
    """
    def __init__(self, config, sim_status):
        self.__config = config
        self.__status = sim_status
        self.__last_curriculum_update = None

    def load_curriculum(self, current_generation:str):
        if 'training' not in self.__config or 'curriculum' not in self.__config['training']:
            return None

        curriculum = self.__config['training']['curriculum']
        if current_generation in curriculum:
            self.__last_curriculum_update = current_generation
            return curriculum[current_generation]

        # comment out the next 2 lines to make tc stateful
        elif self.__last_curriculum_update:
            return curriculum[self.__last_curriculum_update]
        # else:
        return None

    def select_generation(self, current_generation, selection_type=None):
        if selection_type == 'last':
            return current_generation - 1
        elif selection_type == 'random' and current_generation > 1:
            return utils.secure_random.randint(0, current_generation - 1)
        # selects last gen by default
        return current_generation - 1
