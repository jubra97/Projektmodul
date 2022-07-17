import torch

try:
    from rl.envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection
except FileNotFoundError:
    ...

def export_onnx(model, export_path):
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor.actor_target.mu
            print(self.actor)

        def forward(self, observation):
            return self.actor(observation)

    onnxable_model = OnnxablePolicy(model)
    dummy_input = torch.randn(model.observation_space.shape[0])
    onnxable_model.to("cpu")
    torch.onnx.export(onnxable_model, dummy_input, export_path)


def custom_default_json(obj):
    if callable(obj):
        try:
            out = obj.__name__
        except AttributeError:
            out = obj.__class__.__name__
    elif isinstance(obj, DirectControllerOnlineConnection):
        out = None
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} '
                        f'is not JSON serializable')
    return out


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def custom_default(obj):
    if callable(obj):
        try:
            out = obj.__name__
        except AttributeError:
            out = obj.__class__.__name__
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} '
                        f'is not JSON serializable')
    return out