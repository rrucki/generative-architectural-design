# from typing import Dict
#
# import gymnasium as gym
# import torch
# from torch import nn
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#
#
# class SimpleCombinedExtractor(BaseFeaturesExtractor):
#     """
#     Prosty ekstraktor cech, który przetwarza każdą przestrzeň obserwacji
#     małą siecią MLP, a następnie łączy wyniki. Służy jako baseline.
#     """
#
#     def __init__(self, observation_space: gym.spaces.Dict):
#         total_features_dim = 0
#         extractors = {}
#
#         for key, subspace in observation_space.spaces.items():
#             if key == "vertex_vectors":
#                 input_dim = subspace.shape[0] * subspace.shape[1]
#                 feature_dim = 64
#                 extractors[key] = nn.Sequential(
#                     nn.Flatten(),
#                     nn.Linear(input_dim, feature_dim),
#                     nn.ReLU(),
#                 )
#             else:
#                 input_dim = subspace.shape[0]
#                 feature_dim = 32
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(input_dim, feature_dim),
#                     nn.ReLU(),
#                 )
#             total_features_dim += feature_dim
#
#         super().__init__(observation_space, features_dim=total_features_dim)
#         self.extractors = nn.ModuleDict(extractors)
#
#     def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
#         encoded_tensor_list = []
#         for key, extractor in self.extractors.items():
#             tensor = observations[key].float()
#             encoded_tensor_list.append(extractor(tensor))
#
#         return torch.cat(encoded_tensor_list, dim=1)
#
#
# class AttentionFeatureExtractor(BaseFeaturesExtractor):
#     """
#     Ulepszony ekstraktor cech z mechanizmem uwagi.
#     Specjalnie traktuje dane "per-wierzchołek" (wektory, kąty) i "globalne" (pole, szerokość).
#     """
#
#     def __init__(self, observation_space: gym.spaces.Dict):
#         super().__init__(observation_space, features_dim=1)
#
#         extractors = {}
#         feature_dims = {}
#
#         for key, subspace in observation_space.spaces.items():
#             if key == "vertex_vectors":
#                 feature_dim = 128
#                 extractors[key] = nn.Sequential(
#                     nn.Flatten(),
#                     nn.Linear(subspace.shape[0] * subspace.shape[1], feature_dim),
#                     nn.ReLU(),
#                 )
#             elif key == "area_difference":
#                 feature_dim = 32
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_dim),
#                     nn.ReLU(),
#                 )
#             elif key == "width_difference":
#                 feature_dim = 16
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_dim),
#                     nn.ReLU(),
#                 )
#             elif key == "angle_differences":
#                 feature_dim = 8
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_dim),
#                     nn.ReLU(),
#                 )
#             elif key == "remaining_moves":
#                 feature_dim = 4
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_dim),
#                     nn.ReLU(),
#                 )
#             else:
#                 feature_dim = 8
#                 extractors[key] = nn.Sequential(
#                     nn.Linear(subspace.shape[0], feature_dim),
#                     nn.ReLU(),
#                 )
#             feature_dims[key] = feature_dim
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         total_features_dim = sum(feature_dims.values())
#
#         self.attention = nn.MultiheadAttention(
#             embed_dim=total_features_dim,
#             num_heads=4,
#             batch_first=True
#         )
#
#         self._features_dim = total_features_dim
#
#     def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
#         encoded_tensor_list = []
#         for key, extractor in self.extractors.items():
#             tensor = observations[key].float()
#             encoded_tensor_list.append(extractor(tensor))
#
#         concatenated_features = torch.cat(encoded_tensor_list, dim=1)
#
#         attn_input = concatenated_features.unsqueeze(
#             1)
#
#         attn_output, _ = self.attention(attn_input, attn_input, attn_input)
#
#         final_features = attn_output.squeeze(1)
#
#         return final_features