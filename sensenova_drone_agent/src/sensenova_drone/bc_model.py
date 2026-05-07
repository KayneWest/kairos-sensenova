from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:
    class ImageBCPolicy(nn.Module):
        def __init__(
            self,
            num_actions: int,
            hidden_dim: int = 256,
            goal_feature_dim: int = 4,
            frame_stack: int = 1,
        ):
            super().__init__()
            self.goal_feature_dim = int(goal_feature_dim)
            self.frame_stack = max(1, int(frame_stack))
            input_channels = 3 * self.frame_stack
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.goal_mlp = None
            trunk_input_dim = hidden_dim
            if self.goal_feature_dim > 0:
                self.goal_mlp = nn.Sequential(
                    nn.Linear(self.goal_feature_dim, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 32),
                    nn.ReLU(inplace=True),
                )
                trunk_input_dim += 32
            self.trunk = nn.Sequential(
                nn.Flatten(),
                nn.Linear(trunk_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.action_head = nn.Linear(hidden_dim, num_actions)
            self.command_head = nn.Linear(hidden_dim, 5)

        def forward(
            self,
            image: torch.Tensor,
            goal_features: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            features = self.encoder(image)
            features = torch.flatten(features, start_dim=1)
            if self.goal_mlp is not None:
                if goal_features is None:
                    goal_features = torch.zeros(
                        (features.shape[0], self.goal_feature_dim),
                        device=features.device,
                        dtype=features.dtype,
                    )
                goal_embed = self.goal_mlp(goal_features)
                features = torch.cat([features, goal_embed], dim=1)
            features = self.trunk(features)
            return {
                "action_logits": self.action_head(features),
                "command_pred": self.command_head(features),
            }
else:
    class ImageBCPolicy:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate ImageBCPolicy.")
