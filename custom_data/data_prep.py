# import argparse
# from datasets import Dataset, Audio, Value

# parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
# parser.add_argument('--source_data_dir', type=str, required=True, default=False, help='Path to the directory containing the audio_paths and text files.')
# parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')

# args = parser.parse_args()

# scp_entries = open(f"{args.source_data_dir}/audio_paths", 'r').readlines()
# txt_entries = open(f"{args.source_data_dir}/text", 'r').readlines()

# if len(scp_entries) == len(txt_entries):
#     audio_dataset = Dataset.from_dict({"audio": [audio_path.split()[1].strip() for audio_path in scp_entries],
#                     "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]})

#     audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
#     audio_dataset = audio_dataset.cast_column("sentence", Value("string"))
#     audio_dataset.save_to_disk(args.output_data_dir)
#     print('Data preparation done')

# else:
#     print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')

# custom_data/data_prep.py
# custom_data/data_prep.py
# import argparse
# import os
# import glob
# import pandas as pd
# from datasets import Dataset, Audio, Value
# from tqdm import tqdm

# def main():
#     parser = argparse.ArgumentParser(
#         description="從 train-data 讀 CSV，產出 audio_paths、text 兩檔，並存成 HF Dataset"
#     )
#     parser.add_argument(
#         "--train_root", type=str,
#         default=os.path.abspath(
#             os.path.join(os.path.dirname(__file__), "..", "..", "train-data")
#         ),
#         help="train-data 根目錄（預設 ../../train-data）"
#     )
#     parser.add_argument(
#         "--output_data_dir", type=str, default="formatted_data",
#         help="輸出資料夾，裡面會有 audio_paths、text，還有 HF Dataset"
#     )
#     parser.add_argument(
#         "--use_pinyin", action="store_true",
#         help="用客語拼音作 transcript；不加就用客語漢字"
#     )
#     args = parser.parse_args()

#     os.makedirs(args.output_data_dir, exist_ok=True)
#     csv_paths = glob.glob(os.path.join(args.train_root, "*_edit.csv"))
#     if not csv_paths:
#         raise FileNotFoundError(f"在 {args.train_root} 找不到任何 *_edit.csv")

#     audio_paths = []
#     transcripts = []

#     # 進度條：掃描所有 CSV 檔
#     for csv_file in tqdm(csv_paths, desc="Scanning CSV files"):
#         df = pd.read_csv(csv_file, encoding="utf-8")
#         # 進度條：處理每個 CSV 的每一行
#         for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_file)}"):
#             fn = row["檔名"].strip()                # e.g. DM101J2004_001.wav
#             utt = os.path.splitext(fn)[0]         # e.g. DM101J2004_001
#             text = row["客語拼音" if args.use_pinyin else "客語漢字"].strip()

#             # 在 train_root 底下遞迴找 wav
#             wav_candidates = glob.glob(os.path.join(args.train_root, "**", fn), recursive=True)
#             if not wav_candidates:
#                 raise FileNotFoundError(f"找不到音檔 {fn}")
#             wav_path = wav_candidates[0]

#             audio_paths.append((utt, wav_path))
#             transcripts.append((utt, text))

#     # 1) 寫出 audio_paths、text 檔
#     ap_file = os.path.join(args.output_data_dir, "audio_paths")
#     txt_file = os.path.join(args.output_data_dir, "text")
#     with open(ap_file, "w", encoding="utf-8") as fa, \
#          open(txt_file, "w", encoding="utf-8") as ft:
#         for (utt, wav), (_, txt) in zip(audio_paths, transcripts):
#             fa.write(f"{utt} {wav}\n")
#             ft.write(f"{utt} {txt}\n")

#     # 2) 同步生成 HF Dataset
#     print("Generating HuggingFace Dataset…")
#     ds = Dataset.from_dict({
#         "audio": [wav for _, wav in audio_paths],
#         "sentence": [txt for _, txt in transcripts],
#     })
#     ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
#     ds = ds.cast_column("sentence", Value("string"))
#     ds.save_to_disk(os.path.join(args.output_data_dir, "hf_dataset"))

#     print("Done!")
#     print(f" • audio_paths → {ap_file}")
#     print(f" • text        → {txt_file}")
#     print(f" • HF Dataset  → {os.path.join(args.output_data_dir, 'hf_dataset')}")
#     print(f" • 共處理 {len(ds)} 筆資料")

# if __name__ == "__main__":
#     main()
# custom_data/data_prep.py
import argparse
import os
import glob
import pandas as pd
from datasets import Dataset, Audio, Value
from tqdm import tqdm

def infer_dialect_folder(csv_filename):
    """
    根據 CSV 檔名判斷腔調子資料夾（大埔腔 or 詔安腔）
    """
    # 假設 CSV 檔名會包含 '大埔腔' 或 '詔安腔'
    if '大埔腔' in csv_filename:
        return '訓練_大埔腔30H'
    elif '詔安腔' in csv_filename:
        return '訓練_詔安腔30H'
    else:
        raise ValueError(f"無法從 '{csv_filename}' 推斷腔調資料夾名稱")


def main():
    parser = argparse.ArgumentParser(
        description="從 train-data 讀 CSV，產出 audio_paths、text 兩檔，並存成 HF Dataset"
    )
    parser.add_argument(
        "--train_root", type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "train-data")
        ),
        help="train-data 根目錄（預設 ../../train-data）"
    )
    parser.add_argument(
        "--output_data_dir", type=str, default="formatted_data",
        help="輸出資料夾，裡面會有 audio_paths、text，還有 HF Dataset"
    )
    parser.add_argument(
        "--use_pinyin", action="store_true",
        help="用客語拼音作 transcript；不加就用客語漢字"
    )
    args = parser.parse_args()

    os.makedirs(args.output_data_dir, exist_ok=True)
    csv_paths = glob.glob(os.path.join(args.train_root, "*_edit.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"在 {args.train_root} 找不到任何 *_edit.csv")

    audio_paths = []
    transcripts = []

    # 進度條：掃描所有 CSV 檔
    for csv_file in tqdm(csv_paths, desc="Scanning CSV files"):
        csv_name = os.path.basename(csv_file)
        dialect_folder = infer_dialect_folder(csv_name)
        df = pd.read_csv(csv_file, encoding="utf-8")

        # 進度條：處理每個 CSV 的每一行
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_name}"):
            fn = row["檔名"].strip()                # e.g. DM101J2004_001.wav
            utt = os.path.splitext(fn)[0]           # e.g. DM101J2004_001
            text = row["客語拼音" if args.use_pinyin else "客語漢字"].strip()

            # 直接構造 wav 路徑，不再全盤搜索
            speaker = utt.split("_")[0]            # e.g. DM101J2004
            wav_path = os.path.join(
                args.train_root,
                dialect_folder,
                speaker,
                fn
            )
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(f"找不到音檔：{wav_path}")

            audio_paths.append((utt, wav_path))
            transcripts.append((utt, text))

    # 1) 寫出 audio_paths、text 檔
    ap_file = os.path.join(args.output_data_dir, "audio_paths")
    txt_file = os.path.join(args.output_data_dir, "text")
    with open(ap_file, "w", encoding="utf-8") as fa, \
         open(txt_file, "w", encoding="utf-8") as ft:
        for (utt, wav), (_, txt) in zip(audio_paths, transcripts):
            fa.write(f"{utt} {wav}\n")
            ft.write(f"{utt} {txt}\n")

    # 2) 同步生成 HF Dataset
    print("\nGenerating HuggingFace Dataset…")
    ds = Dataset.from_dict({
        "audio": [wav for _, wav in audio_paths],
        "sentence": [txt for _, txt in transcripts],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.cast_column("sentence", Value("string"))
    ds.save_to_disk(os.path.join(args.output_data_dir, "hf_dataset"))

    print("\nDone!")
    print(f" • audio_paths → {ap_file}")
    print(f" • text        → {txt_file}")
    print(f" • HF Dataset  → {os.path.join(args.output_data_dir, 'hf_dataset')}" )
    print(f" • 共處理 {len(ds)} 筆資料")

if __name__ == "__main__":
    main()
