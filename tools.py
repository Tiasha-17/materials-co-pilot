from dotenv import load_dotenv
import os
from mp_api.client import MPRester

load_dotenv()

def get_material(formula: str):
    api_key = os.getenv("MP_API_KEY")

    with MPRester(api_key) as mpr:
        materials = mpr.materials.summary.search(
            formula=formula,
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

    if not materials:
        return {"error": f"No material found for {formula}"}

    material = materials[0]
    return {
        "material_id": str(material.material_id),
        "formula": material.formula_pretty,
        "band_gap": round(material.band_gap, 3),
        "energy_above_hull": round(material.energy_above_hull, 4),
        "crystal_system": str(material.symmetry.crystal_system),
        "space_group": material.symmetry.symbol
    }
