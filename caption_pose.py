import argparse
import os

from tqdm import tqdm

from caption_generator import PROMPT_LONG, generate_caption, load_pose, load_qwen_model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate long camera motion captions from pose files "
            "(e.g., RealEstate10K .txt or TriMotion .json). "
            "Pass one or two paths for quick experiments."
        )
    )
    parser.add_argument(
        "--pose",
        type=str,
        nargs="+",
        required=True,
        help="Pose file path(s): .txt (RealEstate10K) or .json (TriMotion ref_pose).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cam_captions",
        help="Directory to write long caption .txt files (one per pose).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=4,
        help="Frame subsampling interval.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=21,
        help="Number of sampled frames (RealEstate10K .txt).",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=81,
        help="Maximum frames before subsampling (.json poses).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens per caption.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_qwen_model(args.model_name)

    for pose_path in tqdm(args.pose, desc="Generating long captions"):
        camera_pose = load_pose(
            pose_path,
            max_num_frames=args.max_num_frames,
            frame_interval=args.frame_interval,
            num_frames=args.num_frames,
        )
        caption = generate_caption(
            model,
            tokenizer,
            PROMPT_LONG,
            camera_pose,
            max_new_tokens=args.max_new_tokens,
        )

        pose_name = os.path.splitext(os.path.basename(pose_path))[0]
        output_path = os.path.join(args.output_dir, f"{pose_name}_long.txt")
        with open(output_path, "w") as f:
            f.write(caption.strip())
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
