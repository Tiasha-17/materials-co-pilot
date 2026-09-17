from dotenv import load_dotenv
import os
from mp_api.client import MPRester

load_dotenv()
api_key = os.getenv("MP_API_KEY")

print("API key loaded:", bool(api_key))

with MPRester(api_key) as mpr:
    materials = mpr.materials.summary.search(
        formula="TiO2",
        fields=[
    "material_id",
    "formula_pretty",
    "band_gap",
    "energy_above_hull",
    "symmetry"
]
    )
    materials = sorted(
    materials,
    key=lambda material: material.energy_above_hull
)
print("Number of TiO2 results:", len(materials))

for material in materials[:5]:
    print(
        material.material_id,
        material.formula_pretty,
        "Band gap:",
        round(material.band_gap, 3),
        "Energy above hull:",
        round(material.energy_above_hull, 4),
        "Crystal system:",
        material.symmetry.crystal_system,
        "Space group:",
        material.symmetry.symbol
    )