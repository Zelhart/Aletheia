# physiology_interface.py

from typing import Dict, Tuple, Optional
from collections import defaultdict

class PhysiologyInterface:
    """
    Provides a bridge between the voxel-based physiology and the cognitive core.
    Allows for reading physiological states and issuing actuator commands.
    """
    def __init__(self, voxel_mapping):
        self.voxel_mapping = voxel_mapping

    def get_voxel_signals(self) -> Dict[Tuple[int, int, int], Dict[str, float]]:
        """
        Extracts physiological signals from each voxel for cognitive access.
        Returns a dict mapping voxel coordinates to signal dicts.
        """
        signals = {}
        for vc, voxel in self.voxel_mapping.voxel_grid.items():
            signals[vc] = {
                'activation': voxel.activation_level,
                'energy': voxel.energy_level,
                'oxygen': voxel.oxygen_concentration,
                'co2': voxel.co2_concentration,
                'temperature': voxel.temperature,
                'region': voxel.region_id if voxel.region_id else "none"
            }
        return signals

    def get_region_aggregates(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregates physiological signals by organ/region.
        Useful for high-level cognitive appraisal.
        """
        region_data = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(int)

        for vc, voxel in self.voxel_mapping.voxel_grid.items():
            region = voxel.region_id or "none"
            region_data[region]['activation'] += voxel.activation_level
            region_data[region]['energy'] += voxel.energy_level
            region_data[region]['oxygen'] += voxel.oxygen_concentration
            region_data[region]['co2'] += voxel.co2_concentration
            region_data[region]['temperature'] += voxel.temperature
            counts[region] += 1

        aggregates = {}
        for region, data in region_data.items():
            c = counts[region]
            aggregates[region] = {k: (v / c if c > 0 else 0.0) for k, v in data.items()}

        return aggregates

    def apply_actuator_command(self, region: str, energy_increase: float):
        """
        Example actuator: increases energy level across voxels in a region.
        """
        if region not in self.voxel_mapping.organ_voxels:
            return

        for vc in self.voxel_mapping.organ_voxels[region]:
            voxel = self.voxel_mapping.voxel_grid.get(vc)
            if voxel:
                voxel.energy_level = min(1.0, voxel.energy_level + energy_increase)

    def apply_global_temperature_shift(self, delta_temp: float):
        """
        Example actuator: shifts temperature across all voxels.
        """
        for voxel in self.voxel_mapping.voxel_grid.values():
            voxel.temperature = min(42.0, max(30.0, voxel.temperature + delta_temp))
