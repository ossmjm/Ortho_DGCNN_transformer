import json
import struct
from pathlib import Path
import os

def read_gltf_and_bin(gltf_path):
    """Read GLTF and associated binary file."""
    try:
        with open(gltf_path, 'r') as f:
            gltf = json.load(f)
        bin_file_path = Path(gltf_path).parent / gltf['buffers'][0]['uri']
        with open(bin_file_path, 'rb') as f:
            bin_data = f.read()
        return gltf, bin_data
    except Exception as e:
        print(f"Error reading GLTF/BIN files at {gltf_path}: {e}")
        return None, None

def read_accessor_data(gltf, bin_data, accessor_index):
    """Read accessor data from binary buffer."""
    accessor = gltf['accessors'][accessor_index]
    buffer_view = gltf['bufferViews'][accessor['bufferView']]
    byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    count = accessor['count']
    component_type = accessor['componentType']
    type_str = accessor['type']

    type_num_components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    num_components = type_num_components[type_str]

    component_type_formats = {
        5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'
    }
    fmt = component_type_formats[component_type]
    component_size = struct.calcsize(fmt)
    fmt_str = '<' + fmt * num_components

    data = []
    for i in range(count):
        offset = byte_offset + i * num_components * component_size
        bytes_chunk = bin_data[offset: offset + num_components * component_size]
        unpacked = struct.unpack(fmt_str, bytes_chunk)
        # Reduce precision for floats (e.g., 4 decimal places)
        if component_type == 5126:  # FLOAT
            unpacked = tuple(round(x, 4) for x in unpacked)
        data.append(unpacked)
    return data

def convert_gltf_to_custom_json(gltf_path, output_json_path):
    """Convert GLTF to a custom JSON format with reduced size."""
    gltf, bin_data = read_gltf_and_bin(gltf_path)
    if gltf is None or bin_data is None:
        return False

    output = {"teeth": {}, "gums": None}
    mesh_to_name = {}
    for node in gltf.get('nodes', []):
        if 'mesh' in node and 'name' in node:
            mesh_to_name[node['mesh']] = node['name']
    print(f"Processing {gltf_path} - Mesh mappings: {mesh_to_name}")

    valid_teeth = {"31", "32", "33", "34", "35", "36", "37",
                   "41", "42", "43", "44", "45", "46", "47"}

    for mesh_index, mesh in enumerate(gltf['meshes']):
        primitive = mesh['primitives'][0]
        positions = read_accessor_data(gltf, bin_data, primitive['attributes']['POSITION'])
        indices = read_accessor_data(gltf, bin_data, primitive['indices']) if 'indices' in primitive else []
        indices = [i[0] for i in indices]
        faces = [indices[i:i+3] for i in range(0, len(indices), 3)]

        mesh_name = mesh_to_name.get(mesh_index, f"Unnamed_{mesh_index}")
        print(f"Mesh {mesh_name}: {len(positions)} vertices, {len(faces)} faces")

        # Compact storage: Convert tuples to lists for JSON
        vertices = [list(v) for v in positions]
        if mesh_name in valid_teeth:
            output["teeth"][mesh_name] = {"v": vertices, "f": faces}
            print(f"Assigned as tooth: {mesh_name}")
        elif "mandible" in mesh_name.lower() or mesh_name == "0" or "gum" in mesh_name.lower():
            output["gums"] = {"v": vertices, "f": faces}
            print(f"Assigned as gums: {mesh_name}")
        else:
            print(f"Warning: Mesh {mesh_name} not recognized.")

    missing_teeth = valid_teeth - set(output["teeth"].keys())
    if missing_teeth:
        print(f"Warning: Missing teeth: {missing_teeth}")

    # Save compact JSON
    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output, f, separators=(',', ':'))  # Compact JSON
        print(f"Saved JSON to: {output_json_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON to {output_json_path}: {e}")
        return False

def process_all_cases(input_dir, output_dir):
    """Process all cases in the input directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    # Iterate over all case folders (001, 002, etc.)
    for case_dir in sorted(input_path.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.isdigit():
            continue  # Skip non-directory or non-numeric folders

        case_name = case_dir.name
        gltf_file = case_dir / "before_treatment.gltf"
        if not gltf_file.exists():
            print(f"GLTF file not found in {case_dir}. Skipping.")
            continue

        # Define output JSON path: output_dir/case_name/ori/before_treatment.json
        output_json_path = output_path / case_name / "ori" / "before_treatment.json"
        print(f"\nProcessing case {case_name}...")

        success = convert_gltf_to_custom_json(gltf_file, output_json_path)
        if success:
            print(f"Successfully processed case {case_name}")
        else:
            print(f"Failed to process case {case_name}")

# Example usage
input_dir = '//media/osama/sm/gltf sample cases'  # Update with your input directory
output_dir = '/media/osama/sm/Sample_data'  # Update with your output directory
process_all_cases(input_dir, output_dir)