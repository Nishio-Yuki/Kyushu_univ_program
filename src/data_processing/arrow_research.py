from datasets import load_dataset

# Arrowファイルのパス
arrow_file_path = "/home/pj25000162/ku50001814/eval_dataset/in_silico_perturbation/Cop1KO_isp_mouse_tokenize_dataset_v-n1.dataset/data-00000-of-00001.arrow"

# Arrowファイルを読み込む
dataset = load_dataset("arrow", data_files=arrow_file_path)

# データセットのキー（例：train, validationなど）を確認
print(dataset)

# "train" 部分の行数を表示
num_rows = dataset["train"].num_rows
print(f"Number of rows (elements) in the dataset: {num_rows}")

# または同じ意味
print(len(dataset["train"]))
print("input_ids :",dataset["train"]["input_ids"][0])
print("input_ids :",dataset["train"]["organ_major"][0])
print("input_ids :",dataset["train"]["disease"][0])