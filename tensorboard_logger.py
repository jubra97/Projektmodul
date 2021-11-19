from stable_baselines3.common.callbacks import BaseCallback
from SimulationEnvs import FullAdaptivePT2


class TensorboardCallback(BaseCallback):
    def __init__(self, env:FullAdaptivePT2, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        for key, value in self.env.tensorboard_log.items():
            self.logger.record(key, value)
        self.logger.dump(self.num_timesteps)
        return True
