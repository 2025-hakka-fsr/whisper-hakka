# custom_data/split_datasets.py
import argparse
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(
        description="把 HF Dataset 按指定比例切分成 train_dataset_1 和 eval_dataset_1"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="完整 HF Dataset 資料夾，如 formatted_data_pinyin/hf_dataset"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="切分後儲存根目錄，如 formatted_data_pinyin"
    )
    parser.add_argument(
        "--data_ratio", type=float, default=1.0,
        help="要使用的資料比例 (0.0-1.0)，例如 0.5 表示使用前 50% 的資料"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1,
        help="測試集比例，預設 0.1"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子，確保可重現"
    )
    args = parser.parse_args()

    # 1) 載入完整資料集
    ds = load_from_disk(args.input_dir)
    total = len(ds)
    print(f"Loaded dataset with {total} samples")

    # 2) 選取部分資料 (data_ratio)
    if not 0 < args.data_ratio <= 1.0:
        raise ValueError("--data_ratio 必須在 (0.0, 1.0] 之間")
    if args.data_ratio < 1.0:
        use_count = int(total * args.data_ratio)
        ds = ds.shuffle(seed=args.seed).select(range(use_count))
        print(f"Subset dataset to {use_count} samples ({args.data_ratio*100:.1f}%)")

    # 3) train/test 切分
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Split into train ({len(train_ds)}) and eval ({len(eval_ds)})")

    # 4) 儲存到磁碟
    train_ds.save_to_disk(f"{args.output_dir}/train_dataset_sm")
    eval_ds.save_to_disk(f"{args.output_dir}/eval_dataset_sm")
    print("切分完成：")
    print(f" - train_dataset_1: {len(train_ds)} 筆")
    print(f" - eval_dataset_1: {len(eval_ds)} 筆")

if __name__ == "__main__":
    main()
