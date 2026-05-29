import json
import re

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from transformers import AutoModelForCausalLM, AutoTokenizer


class Camera:
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split("] [")
    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "")
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    target_cam_c2w = np.eye(4, dtype=np.float32)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [
        abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]
    ]
    return np.array(ret_poses, dtype=np.float32)


def extrinsics_dict_to_pose_tensor(
    extrinsics,
    max_num_frames=81,
    frame_interval=4,
):
    cam_idx = list(range(max_num_frames))[::frame_interval]
    traj = np.stack([parse_matrix(extrinsics[f"frame{idx}"]) for idx in cam_idx]).transpose(
        0, 2, 1
    )

    c2ws = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.0
        c2w[:3, 3] /= 100
        c2ws.append(c2w)

    cam_params = [Camera(c2w) for c2w in c2ws]
    poses = []
    for i in range(len(cam_params)):
        relative_pose = get_relative_pose([cam_params[0], cam_params[i]])
        poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])

    pose_embedding = torch.stack(poses, dim=0)
    rot = pose_embedding[:, :, :3]
    trans = pose_embedding[:, :, 3].unsqueeze(-1)
    return torch.cat([rot, trans], dim=-1)


def load_pose_from_json(pose_path, max_num_frames=81, frame_interval=4):
    with open(pose_path, "r") as f:
        data = json.load(f)

    if "extrinsics" in data:
        extrinsics = data["extrinsics"]
    else:
        frame_keys = sorted(
            (k for k in data if k.startswith("frame")),
            key=lambda k: int(re.search(r"\d+", k).group()),
        )
        if frame_keys and isinstance(data[frame_keys[0]], str):
            extrinsics = {k: data[k] for k in frame_keys}
        else:
            raise ValueError(
                f"Unsupported JSON pose format: {pose_path}. "
                "Expected {{'extrinsics': {{'frame0': ...}}}} or per-frame strings."
            )

    return extrinsics_dict_to_pose_tensor(
        extrinsics,
        max_num_frames=max_num_frames,
        frame_interval=frame_interval,
    )


def load_pose_from_realestate_txt(
    pose_path,
    frame_interval=4,
    num_frames=21,
):
    with open(pose_path, "r") as f:
        lines = f.readlines()

    extrinsic_values_list = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.strip().split()
        if len(values) >= 19:
            extrinsic_values_list.append([float(v) for v in values[7:19]])

    max_idx = min(
        (num_frames - 1) * frame_interval + 1,
        len(extrinsic_values_list),
    )
    cam_idx = list(range(0, max_idx, frame_interval))
    selected_extrinsics = [extrinsic_values_list[i] for i in cam_idx]

    c2w_matrices = []
    for extrinsic_values in selected_extrinsics:
        matrix_3x4 = np.array(extrinsic_values).reshape(3, 4)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = matrix_3x4
        c2w_matrices.append(np.linalg.inv(w2c))

    cam_params = [Camera(c2w) for c2w in c2w_matrices]
    poses = []
    for i in range(len(cam_params)):
        relative_pose = get_relative_pose([cam_params[0], cam_params[i]])
        poses.append(torch.as_tensor(relative_pose)[:, :3, :][1])

    pose_embedding = torch.stack(poses, dim=0)
    rot = pose_embedding[:, :, :3]
    trans = pose_embedding[:, :, 3].unsqueeze(-1)
    return torch.cat([rot, trans], dim=-1)


def load_pose(pose_path, max_num_frames=81, frame_interval=4, num_frames=21):
    if pose_path.endswith(".json"):
        return load_pose_from_json(
            pose_path,
            max_num_frames=max_num_frames,
            frame_interval=frame_interval,
        )
    if pose_path.endswith(".txt"):
        return load_pose_from_realestate_txt(
            pose_path,
            frame_interval=frame_interval,
            num_frames=num_frames,
        )
    raise ValueError(f"Unsupported pose file: {pose_path} (use .json or .txt)")


def process_poses_for_rdf_coords(camera_poses):
    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.detach().cpu().numpy()

    T = len(camera_poses)
    if camera_poses.shape[-1] == 12:
        camera_poses = camera_poses.reshape(T, 3, 4)

    trans = camera_poses[:, :, 3]
    rots_mat = camera_poses[:, :, :3]

    r = R.from_matrix(rots_mat)
    euler = r.as_euler("xyz", degrees=True)

    indices = np.linspace(0, T - 1, T, dtype=int)
    summary_lines = []

    trans_thresh = 0.02
    rot_thresh = 0.3
    dominant_ratio = 5.0

    total_disp = trans[-1] - trans[0]
    summary_lines.append("Analysis Summary:")
    summary_lines.append(
        f"- Total Displacement: X={total_disp[0]:.2f}m, "
        f"Y={total_disp[1]:.2f}m, Z={total_disp[2]:.2f}m"
    )
    summary_lines.append("\nSequential Motion Phases:")

    for i in range(len(indices) - 1):
        curr_idx = indices[i]
        next_idx = indices[i + 1]

        d_trans = trans[next_idx] - trans[curr_idx]
        d_rot = euler[next_idx] - euler[curr_idx]

        for k in range(3):
            if d_rot[k] > 180:
                d_rot[k] -= 360
            if d_rot[k] < -180:
                d_rot[k] += 360

        max_trans = np.max(np.abs(d_trans))
        max_rot = np.max(np.abs(d_rot))
        phase_desc = []

        if max_trans > trans_thresh:
            if abs(d_trans[0]) > trans_thresh and (
                max_trans / (abs(d_trans[0]) + 1e-6) < dominant_ratio
            ):
                label = "Truck Right" if d_trans[0] > 0 else "Truck Left"
                phase_desc.append(f"{label} ({abs(d_trans[0]):.2f}m)")
            if abs(d_trans[1]) > trans_thresh and (
                max_trans / (abs(d_trans[1]) + 1e-6) < dominant_ratio
            ):
                label = "Pedestal Down" if d_trans[1] > 0 else "Pedestal Up"
                phase_desc.append(f"{label} ({abs(d_trans[1]):.2f}m)")
            if abs(d_trans[2]) > trans_thresh and (
                max_trans / (abs(d_trans[2]) + 1e-6) < dominant_ratio
            ):
                label = "Dolly In" if d_trans[2] > 0 else "Dolly Out"
                phase_desc.append(f"{label} ({abs(d_trans[2]):.2f}m)")

        if max_rot > rot_thresh:
            if abs(d_rot[0]) > rot_thresh and (
                max_rot / (abs(d_rot[0]) + 1e-6) < dominant_ratio
            ):
                phase_desc.append(
                    f"Tilt {'Up' if d_rot[0] > 0 else 'Down'} ({abs(d_rot[0]):.1f}°)"
                )
            if abs(d_rot[1]) > rot_thresh and (
                max_rot / (abs(d_rot[1]) + 1e-6) < dominant_ratio
            ):
                phase_desc.append(
                    f"Pan {'Right' if d_rot[1] > 0 else 'Left'} ({abs(d_rot[1]):.1f}°)"
                )
            if abs(d_rot[2]) > rot_thresh and (
                max_rot / (abs(d_rot[2]) + 1e-6) < dominant_ratio
            ):
                phase_desc.append(
                    f"Roll {'CW' if d_rot[2] > 0 else 'CCW'} ({abs(d_rot[2]):.1f}°)"
                )

        progress = int((i / max(1, len(indices) - 2)) * 100)
        if not phase_desc:
            summary_lines.append(f"Time {progress}%: Stationary")
        else:
            summary_lines.append(f"Time {progress}%: {', '.join(phase_desc)}")

    return "\n".join(summary_lines)


PROMPT_LONG = """You are an expert Cinematographer analyzing camera tracking data.
Your task is to generate a **descriptive yet concise** paragraph describing the camera movement.

Coordinate System (Relative to Start):
- X-axis: Truck (Right/Left) | Y: Pedestal (Down/Up) | Z: Dolly (In/Out)
- Rotation: Pan, Tilt, Roll

Strict Output Guidelines:
1. **No Headers**: Do NOT output "Caption:" or "Here is the description". Start directly with the sentence.
2. **Length**: Write exactly **3 to 5 sentences**.
3. **Focus on Evolution**: Describe how the movement changes over time.
   - Use transition words: "Initially", "Gradually", "Simultaneously", "Towards the end".
   - Describe speed changes: "accelerating", "slowing down", "steady pace".
4. **Compound Motion**: If multiple axes move together, combine them.
   - Example: "The camera dollies in while slowly panning right."
5. **Tone**: Technical, factual, and smooth. Avoid emotional words like "dramatic" or "poetic".

Data Analysis:
"""


PROMPT_SHORT = """You are an expert Video Editor.
I will provide a time-series analysis of camera motion.

Coordinate System (OpenCV Standard, Relative to Initial View):
- X-axis: "Truck" (Right/Left)
- Y-axis: "Pedestal" (Down/Up)
- Z-axis: "Dolly" (In/Out)
- Rotation: Pan, Tilt, Roll.

Your Goal:
Generate a concise camera motion caption (1-2 sentences) that captures the **temporal flow**.

Instructions:
1. **Ignore Noise**: Do not describe minor jitters. Focus on the main action.
2. **Detect Sequence**: If the motion changes direction (e.g., Dolly In -> Pan Right), clearly describe the transition using words like "starts with", "then", "followed by", or "while".
3. **Simultaneous vs. Sequential**:
   - If multiple motions happen at the *same time*, use "while" (e.g., "Trucks right while panning left").
   - If motions happen *one after another*, use "then" (e.g., "Dolly in, then tilts up").
4. **No Numbers**: Do not output meters or degrees.
5. **Velocity vs Position**: The data provided is the *displacement per step* (speed), NOT the absolute position.
   - If "Pedestal Up" decreases from 0.06m to 0.01m, it means the camera is **still moving up, but slowing down**. It is NOT moving down.
   - A direction change only happens if the label changes (e.g., from "Pedestal Up" to "Pedestal Down").
Data Analysis:
"""


def load_qwen_model(model_name="Qwen/Qwen3-4B-Instruct-2507"):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    print("Model loaded.")
    return model, tokenizer


def generate_caption(model, tokenizer, prompt, camera_pose, max_new_tokens=128):
    data_analysis = process_poses_for_rdf_coords(camera_pose)
    text_input = prompt + data_analysis
    messages = [{"role": "user", "content": text_input}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)
