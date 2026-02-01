"""metatrain/experimental/upet/model.py

Minimal UPET model for torch.compile compatibility with metatrain.
"""

import torch
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path

from metatensor.torch import TensorMap, TensorBlock, Labels
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

from .documentation import ModelHypers
from .structures import systems_to_upet_batch
from .upet import UPET, predict


class UPETModel(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        self.atomic_types = list(dataset_info.atomic_types)
        
        outputs_capabilities = {}
        for target_name, target_info in dataset_info.targets.items():
            outputs_capabilities[target_name] = ModelOutput(
                unit=target_info.unit,
                per_atom=False,
            )

        self.capabilities = ModelCapabilities(
            outputs=outputs_capabilities,
            atomic_types=self.atomic_types,
            interaction_range=hypers["cutoff"],
            length_unit=dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype="float64",
        )

        self.model = UPET(
            d_pet=hypers["d_pet"],
            d_node=hypers["d_node"],
            d_head=hypers["d_head"],
            d_feedforward=hypers["d_feedforward"],
            num_heads=hypers["num_heads"],
            num_attention_layers=hypers["num_attention_layers"],
            num_gnn_layers=hypers["num_gnn_layers"],
            cutoff=hypers["cutoff"],
            cutoff_width=hypers["cutoff_width"],
            normalization=hypers["normalization"],
            activation=hypers["activation"],
            transformer_type=hypers["transformer_type"],
            attention_temperature=hypers["attention_temperature"],
            featurizer_type=hypers["featurizer_type"],
            num_species=len(self.atomic_types),
        )
        
        self.species_map = {z: i for i, z in enumerate(self.atomic_types)}

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return {
            target_name: ModelOutput(
                unit=target_info.unit,
                per_atom=False,
            )
            for target_name, target_info in self.dataset_info.targets.items()
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        model_dtype = next(self.model.parameters()).dtype
        
        compute_forces = False
        compute_stress = False
        
        for output_options in outputs.values():
            if hasattr(output_options, 'explicit_gradients'):
                gradients = output_options.explicit_gradients
                if gradients is not None:
                    if "positions" in gradients:
                        compute_forces = True
                    if "strain" in gradients:
                        compute_stress = True
        
        batch = systems_to_upet_batch(
            systems,
            cutoff=self.hypers["cutoff"],
            cutoff_width=self.hypers.get("cutoff_width", 0.5),
            species_to_index=self.species_map,
            device=device,
            dtype=model_dtype,
            num_neighbors_adaptive=self.hypers.get("num_neighbors_adaptive"),
        )
        
        raw_results = predict(
            self.model,
            batch,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
        )
        
        results: Dict[str, TensorMap] = {}
        n_systems = len(systems)
        
        for target_name in outputs.keys():
            energy_values = raw_results["energy"]
            
            if energy_values.dim() == 1:
                energy_values = energy_values.unsqueeze(-1)
            
            samples = Labels(
                names=["system"],
                values=torch.arange(n_systems, device=device, dtype=torch.int32).reshape(-1, 1),
            )
            
            properties = Labels(
                names=["energy"],
                values=torch.zeros((1, 1), device=device, dtype=torch.int32),
            )
            
            block = TensorBlock(
                values=energy_values,
                samples=samples,
                components=[],
                properties=properties,
            )
            
            if compute_forces and "forces" in raw_results:
                forces = raw_results["forces"]
                
                grad_sample_list = []
                for s_idx, system in enumerate(systems):
                    n_atoms = len(system)
                    for a_idx in range(n_atoms):
                        grad_sample_list.append([s_idx, a_idx])
                
                grad_samples = Labels(
                    names=["sample", "atom"],
                    values=torch.tensor(grad_sample_list, device=device, dtype=torch.int32),
                )
                
                grad_values = -forces.unsqueeze(-1)
                
                grad_block = TensorBlock(
                    values=grad_values,
                    samples=grad_samples,
                    components=[
                        Labels(
                            names=["xyz"],
                            values=torch.arange(3, device=device, dtype=torch.int32).reshape(-1, 1),
                        )
                    ],
                    properties=properties,
                )
                block.add_gradient("positions", grad_block)
            
            if compute_stress and "stress" in raw_results:
                stress = raw_results["stress"]
                
                grad_samples = Labels(
                    names=["sample"],
                    values=torch.arange(n_systems, device=device, dtype=torch.int32).reshape(-1, 1),
                )
                
                grad_values = stress.unsqueeze(-1)
                
                grad_block = TensorBlock(
                    values=grad_values,
                    samples=grad_samples,
                    components=[
                        Labels(
                            names=["xyz_1"],
                            values=torch.arange(3, device=device, dtype=torch.int32).reshape(-1, 1),
                        ),
                        Labels(
                            names=["xyz_2"],
                            values=torch.arange(3, device=device, dtype=torch.int32).reshape(-1, 1),
                        ),
                    ],
                    properties=properties,
                )
                block.add_gradient("strain", grad_block)
            
            keys = Labels(
                names=["_"],
                values=torch.zeros((1, 1), device=device, dtype=torch.int32),
            )
            results[target_name] = TensorMap(keys=keys, blocks=[block])
        
        return results

    def restart(self, dataset_info: DatasetInfo) -> "UPETModel":
        self.dataset_info = dataset_info
        self.atomic_types = list(dataset_info.atomic_types)
        self.species_map = {z: i for i, z in enumerate(self.atomic_types)}
        return self

    def get_checkpoint(self) -> Dict[str, Any]:
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        return {
            "model_state_dict": model_to_save.state_dict(),
            "model_hypers": dict(self.hypers),
            "dataset_info": self.dataset_info,
            "model_ckpt_version": self.__checkpoint_version__,
            "architecture_name": "experimental.upet",
        }

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "UPETModel":
        model = cls(
            hypers=checkpoint["model_hypers"],
            dataset_info=checkpoint["dataset_info"],
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if "architecture_name" not in checkpoint:
            checkpoint["architecture_name"] = "experimental.upet"
        return checkpoint

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        """Export model to AtomisticModel format.
        
        Note: torch.compile is not compatible with TorchScript export.
        The model will be exported without compilation.
        """
        # Get the uncompiled model
        model_to_export = self.model
        if hasattr(self.model, "_orig_mod"):
            # Restore uncompiled model for export
            self.model = self.model._orig_mod
        
        dtype = next(self.model.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"Unsupported dtype {dtype} for UPET export")

        self.to(dtype)
        self.eval()

        capabilities = ModelCapabilities(
            outputs=self.capabilities.outputs,
            atomic_types=self.capabilities.atomic_types,
            interaction_range=self.hypers["cutoff"],
            length_unit=self.capabilities.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.metadata, metadata)
        
        return AtomisticModel(self, metadata, capabilities)
