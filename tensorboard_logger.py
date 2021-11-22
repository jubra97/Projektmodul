from stable_baselines3.common.callbacks import BaseCallback
from envs.DirectControllerPT2 import DirectControllerPT2


class TensorboardCallback(BaseCallback):
    def __init__(self, env:DirectControllerPT2, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        for key, value in self.env.tensorboard_log.items():
            self.logger.record(key, value)
        return True
