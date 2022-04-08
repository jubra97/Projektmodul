import torch
import stable_baselines3.common.preprocessing
from envs.DirectControllerOnlineConnection import DirectControllerOnlineConnection


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