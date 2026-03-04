import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


def load_map_config(map_name: str, specs_dir: str) -> dict:
    """
    Load the map configuration from a YAML file by map name.

    Args:
        map_name (str): The name of the map.
        specs_dir (str): Directory containing map information files.

    Returns:
        dict: The map configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    file_path = Path(specs_dir) / f"{map_name}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"map config file not found: {file_path}")

    with file_path.open("r") as file:
        return yaml.safe_load(file) or {}


def get_map_info_util(name: str) -> dict:
    """
    Get the map information in a more accessible format.

    Args:
        name (str): The name of the map.

    Returns:
        dict: Get map information with map name as key.
    """
    # Resolve relative to the project root (one level up from utils)
    specs_dir = Path(__file__).parent.parent / "map_specifications"

    name = name.replace(" ", "_")
    config = load_map_config(name, str(specs_dir))
    parsed_config = {}

    # Check if the loaded config has the required fields
    if not config:
        raise ValueError(f"No configuration found for map '{name}'")

    # Check required fields
    for field in ("type", "locations"):
        if field not in config or config[field] in (None, ""):
            raise ValueError(f"Map '{name}' is missing required field: {field}")

    # Create configuration with robot name as key
    parsed_config[name] = {"type": config["type"], "locations": config["locations"]}

    return parsed_config


def get_map_info_list_util() -> dict:
    """
    Get a list of all available map specification files.

    Returns:
        dict: List of available map names that can be used with get_map_info_util.
    """
    # Resolve relative to the project root (one level up from utils)
    specs_path = Path(__file__).parent.parent / "map_specifications"

    if not specs_path.exists():
        return {"error": f"Map specifications directory not found: {specs_path}"}

    try:
        # Find all YAML files in the specifications directory
        yaml_files = list(specs_path.glob("*.yaml"))

        if not yaml_files:
            return {"error": "No map specification files found"}

        # Extract robot names (file names without .yaml extension)
        map_names = [file.stem for file in yaml_files]
        map_names.sort()  # Sort alphabetically for consistency

        return {"map_specifications": map_names, "count": len(map_names)}

    except Exception as e:
        return {"error": f"Failed to read map specifications directory: {str(e)}"}


def write_map_location_util(
    map_name: str,  # e.g. "hospital", "office"
    name: str,  # e.g. "AED"
    description: str,
    x: float,
    y: float,
    yaw: float,
) -> dict:
    """
    Add or update a location entry in map_specifications/{map_name}.yaml.

    Args:
        map_name (str): Map name without extension (e.g., 'hospital', 'office').
        name (str): Location name (e.g., 'AED').
        description (str): Description of the location.
        x (float): X coordinate.
        y (float): Y coordinate.
        yaw (float): Orientation in radians.

    Returns:
        dict: Status, which file was modified, and the added/updated location.
    """
    try:
        # 1) Resolve specs directory (same as get_map_info_util)
        specs_dir = Path(__file__).parent.parent / "map_specifications"

        if not specs_dir.exists():
            return {"error": f"Map specifications directory not found: {specs_dir}"}

        # 2) Build the YAML file path for the selected map
        #    Example: hospital -> map_specifications/hospital.yaml
        safe_map_name = map_name.replace(" ", "_")
        map_file = specs_dir / f"{safe_map_name}.yaml"

        if not map_file.exists():
            return {"error": f"Map file not found: {map_file}"}

        # 3) Load existing YAML
        with map_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 4) Ensure 'locations' exists
        if "locations" not in data or data["locations"] is None:
            data["locations"] = []

        locations = data["locations"]

        # 5) Build the new/updated location entry
        new_location = {
            "name": name,
            "description": description,
            "pose": {
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
            },
        }

        # 6) If a location with the same name exists → update it, else append
        action = "added"
        for idx, loc in enumerate(locations):
            if loc.get("name") == name:
                locations[idx] = new_location
                action = "updated"
                break
        else:
            locations.append(new_location)

        data["locations"] = locations

        # 7) Write back to the same YAML file
        with map_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        return {
            "status": "success",
            "action": action,
            "map_name": safe_map_name,
            "file": str(map_file),
            "location": new_location,
        }

    except Exception as e:
        return {"error": f"Failed to write semantic location: {str(e)}"}


def draw_robot_pose_util(
    map_message: Dict[str, Any],
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    arrow_length_m: float = 0.5,
) -> Dict[str, Any]:
    """
    Draw the robot's current position and heading on the map image.

    Loads received_map_overlay.png (axes already drawn) if it exists, otherwise
    falls back to received_map.png.  Converts world-frame coordinates to pixel
    coordinates and draws an orange filled circle at the robot's position and an
    orange arrow indicating its heading.  The result is saved as
    ./map/received_map_robot.png.

    Args:
        map_message (dict): OccupancyGrid message (or dict with a 'msg' key)
            containing an 'info' sub-dict with width, height, resolution,
            and origin.
        robot_x (float): Robot X position in the map frame (meters).
        robot_y (float): Robot Y position in the map frame (meters).
        robot_yaw (float): Robot heading in radians (0 = facing +X axis).
        arrow_length_m (float): Heading arrow length in meters. Default 0.5.

    Returns:
        dict: On success:
            {
                "robot_map_path": str,
                "robot_world_position": {"x": float, "y": float, "yaw_rad": float},
                "pixel_position": {"col": int, "row": int},
                "source_image": str,
                "message": str,
            }
            On failure: {"error": str}
    """
    try:
        msg = map_message.get("msg", map_message)
        info = msg.get("info", {})

        height = info.get("height")
        resolution = info.get("resolution")
        origin_pos = info.get("origin", {}).get("position", {})

        if height is None or resolution is None:
            return {"error": "Missing height or resolution in map metadata."}

        height = int(height)
        resolution = float(resolution)
        origin_x = float(origin_pos.get("x", 0.0))
        origin_y = float(origin_pos.get("y", 0.0))

        # Load the best available map image.
        overlay_path = os.path.join("./map", "received_map_overlay.png")
        base_path = os.path.join("./map", "received_map.png")

        if os.path.exists(overlay_path):
            img = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
            source_path = overlay_path
        elif os.path.exists(base_path):
            img = cv2.imread(base_path, cv2.IMREAD_COLOR)
            source_path = base_path
        else:
            return {"error": "No map image found. Call analyze_previously_received_map first."}

        if img is None:
            return {"error": f"Failed to load map image from: {source_path}"}

        # Ensure the image is BGR (parse_map saves grayscale PNG).
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_h, img_w = img.shape[:2]

        # World → pixel (no rotation applied).
        #   col = (rx - origin_x) / resolution       (increases rightward)
        #   row = H - (ry - origin_y) / resolution   (increases downward; Y flipped)
        robot_col = int(round((robot_x - origin_x) / resolution))
        robot_row = int(round(float(height) - (robot_y - origin_y) / resolution))
        robot_col = max(0, min(img_w - 1, robot_col))
        robot_row = max(0, min(img_h - 1, robot_row))

        # Build a filled arrow polygon in the robot's local frame.
        #
        # Local axes:  lx = robot forward,  ly = robot left
        # Image mapping from local offset (lx, ly):
        #   Δcol =  lx*cos(yaw) - ly*sin(yaw)
        #   Δrow = -(lx*sin(yaw) + ly*cos(yaw))
        #
        # Arrow shape (lx forward, ly left), scaled by S pixels:
        #
        #         tip
        #          *
        #         /|\
        #        / | \
        #  L-sh *  |  * R-sh   ← arrowhead (wide)
        #        \  |  /
        #    L-b  * | * R-b    ← body shoulders
        #         | | |
        #    L-r  *   * R-r    ← rear
        #
        S = float(max(int(arrow_length_m / resolution), 12))
        local_pts = np.array(
            [
                [ 1.0,  0.0],  # tip (front)
                [-0.6,  0.6],  # rear-left
                [-0.2,  0.0],  # rear notch
                [-0.6, -0.6],  # rear-right
            ],
            dtype=np.float64,
        ) * S

        cos_a = math.cos(robot_yaw)
        sin_a = math.sin(robot_yaw)
        pixel_pts = []
        for lx, ly in local_pts:
            dc = lx * cos_a - ly * sin_a
            dr = -(lx * sin_a + ly * cos_a)
            pixel_pts.append([robot_col + int(round(dc)), robot_row + int(round(dr))])

        pts = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Filled orange arrow with a dark outline for contrast.
        cv2.fillPoly(img, [pts], color=(0, 165, 255))       # BGR orange fill
        cv2.polylines(img, [pts], isClosed=True, color=(0, 80, 180), thickness=1)  # outline

        robot_map_path = os.path.join("./map", "received_map_robot.png")
        os.makedirs("./map", exist_ok=True)
        success = cv2.imwrite(robot_map_path, img)
        if not success:
            return {"error": f"Failed to save robot map image at: {robot_map_path}"}

        return {
            "robot_map_path": robot_map_path,
            "robot_world_position": {
                "x": robot_x,
                "y": robot_y,
                "yaw_rad": robot_yaw,
            },
            "pixel_position": {"col": robot_col, "row": robot_row},
            "source_image": source_path,
            "message": "Robot pose drawn successfully on the map.",
        }

    except Exception as e:
        return {"error": f"Exception while drawing robot pose: {e}"}


def draw_map_axes_util(
    map_message: Dict[str, Any],
    line_thickness: int = 2,
) -> Dict[str, Any]:
    """
    Overlay coordinate axes and a world-aligned grid on a map image
    using OccupancyGrid metadata.

    Args:
        map_message (dict):
            A dictionary that must contain:
            - an 'info' field with:
                - 'width' (int): map width in pixels.
                - 'height' (int): map height in pixels.
                - 'resolution' (float): map resolution in meters per pixel.
                - 'origin.position.x' (float): origin X in map frame (meters).
                - 'origin.position.y' (float): origin Y in map frame (meters).
            The dictionary may optionally be wrapped inside a top-level 'msg' field.

        line_thickness (int):
            Thickness of the axis lines in image pixels. Larger values make the
            rendered axes visually thicker and easier to see on high-resolution maps.

    Returns:
        dict:
            On success:
                {
                    "annotated_map_path": "<path to PNG with axes and grid>",
                    "width": <int>,
                    "height": <int>,
                    "resolution": <float>,
                    "axis_length_m": <float>,
                    "grid_spacing_m": <float>,
                    "world_bounds": {
                        "x": [<float>, <float>],
                        "y": [<float>, <float>]
                    }
                    "message": "Axes and grid drawn successfully."
                }

            On failure:
                {
                    "error": "<reason string>"
                }
    """
    try:
        # Allow both {"msg": {...}} or direct message dict
        msg = map_message.get("msg", map_message)
        if not isinstance(msg, dict):
            return {"error": "Invalid map_message: expected a dict or a dict with a 'msg' field."}

        info = msg.get("info", {})

        # Use fixed path as you designed earlier
        map_image_path = os.path.join("./map", "received_map.png")

        width = info.get("width")
        height = info.get("height")
        resolution = info.get("resolution")
        origin_pos = info.get("origin", {}).get("position", {})

        if width is None or height is None or resolution is None:
            return {"error": "Missing width, height, or resolution in map metadata."}

        # Load the base map image
        img = cv2.imread(map_image_path, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"Failed to load map image from path: {map_image_path}"}

        h, w = img.shape[:2]

        # If the actual image size differs from the metadata, prefer the image size
        if w != int(width) or h != int(height):
            print(
                f"[draw_map_axes] Warning: image size ({w}x{h}) does not match "
                f"metadata ({width}x{height}). Using image size for drawing.",
                file=sys.stderr,
            )
            width = w
            height = h

        # Compute the physical map size in meters
        map_width_m = float(width) * float(resolution)
        map_height_m = float(height) * float(resolution)

        # Automatically determine axis length as 20% of the shorter map dimension
        shortest_side_m = min(map_width_m, map_height_m)
        axis_length_m = max(shortest_side_m * 0.2, resolution)  # at least one pixel
        axis_length_px = max(int(axis_length_m / float(resolution)), 1)

        # Compute image pixel of world (0,0)
        grid_origin_x = float(origin_pos.get("x", 0.0))
        grid_origin_y = float(origin_pos.get("y", 0.0))

        # World(0,0) → image pixel(u,v)
        # u increases to the right, v increases downward in image space.
        origin_u = int(round((0.0 - grid_origin_x) / float(resolution)))
        origin_v = int(round(height - (0.0 - grid_origin_y) / float(resolution)))

        # Clamp within image boundaries
        origin_u = max(0, min(width - 1, origin_u))
        origin_v = max(0, min(height - 1, origin_v))

        # Choose a "nice" grid spacing in meters so that the number of lines is reasonable.
        # Target ~10-20 grid cells across the shortest side.
        target_cells = 15.0
        base_spacing = max(shortest_side_m / target_cells, resolution)

        # Snap to a human-friendly spacing (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, ...)
        nice_steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        grid_spacing_m = nice_steps[-1]
        for step in nice_steps:
            if step >= base_spacing:
                grid_spacing_m = step
                break

        # World extents covered by the map
        x_min = grid_origin_x
        x_max = grid_origin_x + map_width_m
        y_min = grid_origin_y
        y_max = grid_origin_y + map_height_m

        # Vertical grid lines (constant x = k * grid_spacing_m)
        # Find k range such that lines fall inside [x_min, x_max]
        if grid_spacing_m > 0:
            k_start_x = math.ceil(x_min / grid_spacing_m)
            k_end_x = math.floor(x_max / grid_spacing_m)

            for k in range(k_start_x, k_end_x + 1):
                x_world = k * grid_spacing_m
                u = (x_world - grid_origin_x) / float(resolution)
                u_int = int(round(u))
                if 0 <= u_int < width:
                    cv2.line(
                        img,
                        (u_int, 0),
                        (u_int, height - 1),
                        color=(200, 200, 200),  # light gray
                        thickness=1,
                    )

            # Horizontal grid lines (constant y = k * grid_spacing_m)
            k_start_y = math.ceil(y_min / grid_spacing_m)
            k_end_y = math.floor(y_max / grid_spacing_m)

            for k in range(k_start_y, k_end_y + 1):
                y_world = k * grid_spacing_m
                v = height - (y_world - grid_origin_y) / float(resolution)
                v_int = int(round(v))
                if 0 <= v_int < height:
                    cv2.line(
                        img,
                        (0, v_int),
                        (width - 1, v_int),
                        color=(200, 200, 200),  # light gray
                        thickness=1,
                    )

        # +X axis endpoint
        x_end_u = min(width - 1, origin_u + axis_length_px)
        x_end_v = origin_v

        # +Y axis endpoint
        y_end_u = origin_u
        y_end_v = max(0, origin_v - axis_length_px)

        # Draw +X axis (red)
        cv2.arrowedLine(
            img,
            (origin_u, origin_v),
            (x_end_u, x_end_v),
            color=(0, 0, 255),
            thickness=line_thickness,
            tipLength=0.05,
        )

        # Draw +Y axis (green)
        cv2.arrowedLine(
            img,
            (origin_u, origin_v),
            (y_end_u, y_end_v),
            color=(0, 255, 0),
            thickness=line_thickness,
            tipLength=0.05,
        )

        # Mark world(0,0) with a small blue circle
        cv2.circle(
            img,
            (origin_u, origin_v),
            radius=4,
            color=(255, 0, 0),
            thickness=-1,
        )

        # Save annotated map
        base, ext = os.path.splitext(map_image_path)
        annotated_path = base + "_overlay" + (ext or ".png")
        os.makedirs(os.path.dirname(annotated_path) or ".", exist_ok=True)

        saved = cv2.imwrite(annotated_path, img)
        if not saved:
            return {"error": f"Failed to save annotated image to: {annotated_path}"}

        return {
            "annotated_map_path": annotated_path,
            "width": int(width),
            "height": int(height),
            "resolution": float(resolution),
            "axis_length_m": float(axis_length_m),
            "grid_spacing_m": float(grid_spacing_m),
            "world_bounds": {
                "x": [float(x_min), float(x_max)],
                "y": [float(y_min), float(y_max)],
            },
            "message": "Axes and grid drawn successfully.",
        }

    except Exception as e:
        return {"error": f"Exception while drawing axes and grid: {e}"}
