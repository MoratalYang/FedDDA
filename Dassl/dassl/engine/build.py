from Dassl.dassl.utils import Registry, check_availability
from trainers.FEDDDA import FEDDDA


TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(FEDDDA)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
