
import torch
from torch_geometric.data import Data, Batch
from fairchem.core.datasets import data_list_collater

from .base_potential import BasePotential, PotentialOutput

class UMAPotential(BasePotential):
    def __init__(self, model_path, dataset, **kwargs):
        """
        Constructor for UMA Potential

        Parameters
        ----------
        model_path: str
            path to the model. eg. 'weights/uma/model.pt'
        dataset: str
            dataset name. eg. 'oc20'
        """
        super().__init__(**kwargs)
        self.model = self.load_model(model_path)
        self.dataset = dataset
        self.n_eval = 0


    def forward(self, positions):
        data = self.data_formatter(positions)
        pred = self.model.predict(data)
        self.n_eval += 1
        energies = pred.get(f'{self.dataset}_energy').unsqueeze(-1)
        forces = pred.get(f'{self.dataset}_forces')
        forces = forces.view(*positions.shape)
        return PotentialOutput(energies=energies, forces=forces)


    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        return model

    def data_formatter(self, positions):
        positions = positions.view(*positions.shape[:-1], self.n_atoms, 3)
        cell = self.cell.unsqueeze(0)
        atomic_numbers = self.atomic_numbers
        natoms = self.n_atoms
        pbc = self.pbc
        spin = 0
        charge = 0
        dataset = self.dataset

        data_list = []
        for pos in positions:
            data = Data(
                pos=pos,
                cell=cell,
                atomic_numbers=atomic_numbers,
                natoms=natoms,
                pbc=pbc,
                spin=spin,
                charge=charge,
                dataset=dataset,
            )
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        print(batch)
        
        return batch