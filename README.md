# 九州大学共同研究 引き継ぎREADME

九州大学で行われた遺伝子勉強会の引き継ぎ用テキストです．多分一通り書いていると思いますが，エラー等が出たら連絡をください．また，これらのプログラムはMouse-Geneformerの概要や，共同研究の内容をざっくりと知っている前提で作成しているため，知らない場合は少し勉強しておくと良いかもしれません（プログラム自体は動かせますが，何をしているのかわからないかも）．

九州大学側の意図として，事前学習プログラムは動かさずに，in silico摂動実験等にMouse-Geneformerを使えればOKとのことなので，主にin silico摂動実験を動かすことや，UMAPを可視化することによる分析を目的としています．

- 環境構築
    - `geneformer.def` から sifイメージを作成して，singularity上で実験してください
- raw_dataから学習データを作成する方法
    - raw_dataの収集方法は，`document`フォルダに入っているPDFを見てください．現状は`10x Genomics`のみですが今後追加予定です．
    - 九州大学での勉強会では，副腎皮質に関するraw_dataを使用しました．Quality Control (QC) フィルタリングと，Rank value encodingを行ってからGeneformerに入力可能な形式に変換する必要があります．
    - `src/data_processing/data_processing.py`の下記のパスを変更
    
    ```python
    INPUT_DIR        = "/path/to/h5/folder"      # .h5 / .h5ad が入っているフォルダ
    OUTPUT_DIR       = "/path/to/output/folder"  # save_to_disk 先
    VAR_GENEKEY      = "genekey"          # var の最終キー（例: 'gene_symbol','gene','features','gene_ids','ensembl_id'）
    USE_RAW          = True                # geneの抽出・順位付けに raw を使うか
    
    MAPPING_PICKLE   = "/path/to/symbol/to/ensemble"
    
    ENSEMBL2ID_PATH  = "/path/to/ensemble/to/ID"
    ```
    
    - これらを変更して，`python3 data_processing.py`を実行すればOKです．ただ，ここで得られるデータ形式としては，`input_ids`，`length`の2カラムです．`input_ids`は，遺伝子配列を降順にソートした数値ID，`length`は`input_ids`の長さです．in silico摂動実験を行う際はこの他にもいくつかカラムが必要になります．そのため，このプログラムで作成したデータに対してカラムを追加する必要があります．
- 作成したデータにカラムを追加する方法
    - `src/data_processing/add_column.ipynb`の以下のパスを変更して上から順に実行してください（正直jupyterはsingularityで使いづらいので.pyに変更する予定です）．
    
    ```python
    INPUT_DIR = "/path/to/arrow/data" # data_processing.pyで作成したarrowが入っているフォルダのパスを指定
    OUTPUT_DIR = "/path/to/output/folder" # カラム追加後のデータを出力するフォルダのパスを指定 
    
    NEW_COLUMNS = {
        "disease": "MC",
        "cell_types" : "nan",
        "organ_major" : "nan",
    }
    ```
    
    - `INPUT`，`OUTPUT`は各々の環境に合わせて設定すればOKですが，`NEW_COLUMNS`については結構複雑なので，口頭で説明します．そのため，ここまでできたら西尾に声をかけてください
    - ここまでで in silico摂動実験を行うためのデータ作成は一通り終了です．ただ，in silico摂動実験を行う際は作成したデータを一つのフォルダにまとめる必要があるため，`/src/data_processing/merge_multi_arrow.py`を実行します．以下のパスを変更して実行してください．
    
    ```python
    INPUT_DIRS = [
    		"/path/to/folder1",
    		"/path/to/folder2",
    		...
        # 必要に応じて追加
    ]
    OUTPUT_DIR = "/path/to/output/folder"
    ```
    
    - `INPUT_DIRS`は，先ほどの段階までで作成したものを全て指定すればOKです．`OUTPUT_DIR`には出力先のフォルダパスを指定してください．ここまでやれば in silico摂動実験のプログラムに行きましょう．
- in silico摂動実験の実行方法
    - `/src/Mouse-Geneformer/in_silico_perturbation.ipynb`の下記のパスを変更して上から順番に実行してください
    - 以下を変更して最後のセルまで実行すればcsvファイルとして指定したフォルダ以下にin silico摂動実験の結果が出力されます．
    
    ```python
    # ==== 1) ユーザ設定 ====
    # Cell 1
    REPO_ROOT = "/path/to/geneformer/"  # geneformerの親ディレクトリを指定
    PATH_TOKEN_DICT   = Path("/path/to/MLM-re_token_dictionary_v1.pkl")
    PATH_GENE_NAME2ID = Path("/path/to/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl")
    ```
    
    ```python
    # Cell 3
    
    import os
    RESULT_DIR_NAME = "/path/to/result/folder" # resultを保存しておくディレクトリを指定
    # If you choice "CellClassifier", you choice fine tuning model. If you choice "Pretrained", you choice pretrained model.
    
    PRE_TRAIN_MODEL_DIR = "/path/to/mouse-geneformer_pretrain_model" # Mouse-Geneformerの事前学習済みモデルを指定
    DATASET_NAME = "/path/to/in_silico/data" # in silico摂動実験を行うデータのパスを指定
    ```
    
    ```python
    # Cell 11
    
    FINETUNED_MODEL_DIR = "/path/to/FT_model"
    select_perturb_type="delete" # in silico摂動実験の種類を指定
    
    start_state = 'MC' # 開始状態を指定
    end_state = 'FO' # 目標状態を指定
    ```
    
- 仮想シングルセルデータの作成方法
    - 九州大学が今取り組んでいる内容として，バルク解析のデータを仮想的にシングルセルデータに変換するというものがあります．このような仮想シングルセルデータを作成するには，`/src/data_processing/create_virtual_scRNA_data.py`を使用します．下記のパスを変更して実行してください．
    
    ```python
    # 入力（1列目=遺伝子名、2列目以降=サンプル列）
    BULK_MATRIX_PATH: str = "/path/to/bulk/data"  # .csv/.tsv/.txt/.xlsx/.xls すべて対応
    GENE_TO_ENSEMBL_PICKLE: str = "/path/to/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"  # {symbol -> ensembl_id}
    ENSEMBL_TO_TOKEN_PICKLE: str = "/path/to/MLM-re_token_dictionary_v1.pkl"  # {ensembl_id -> int token}
    
    # 出力
    OUTPUT_DIR: str = "/path/to/output_dir" # arrowを出力したいフォルダパスを指定
    ```
    
- UMAP可視化方法
    - 特定のデータについてUMAPを可視化する方法と，in silico摂動実験の前，後，目標状態の3状態でUMAPを可視化する方法の2種類があります．
        - 特定のデータについてUMAPを可視化する方法
            - `/src/Mouse-Geneformer/geneformer/tokenizer.py`の`GENE_MEDIAN_FILE`と`TOKEN_DICTIONARY_FILE`をそれぞれ，`MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl`，`MLM-re_token_dictionary_v1.pkl`に変更してください．
            - `/src/Mouse-Geneformer/disease_type_extract_and_plot_cell_embedding.ipynb` の下記のパスを変更して上から順に実行してください．
            
            ```python
            # Cell 2
            dataset_name = "/path/to/dataset"   # UMAPを可視化したいデータのパスを指定
            ```
            
            ```python
            # Cell 3
            FINETUNED_MODEL_DIR = "/path/to/fine-tuning/model" # in silicoのプログラムで作成したファインチューニングモデルを指定
            
            import os
            DIR_NAME = "/path/to/out_dir" # UMAPを出力するフォルダのパスを指定
            if not os.path.exists(DIR_NAME):
                os.mkdir(DIR_NAME)
            ```
            
        - 3状態でのUMAPを可視化する方法
            - `/src/Mouse-Geneformer/Umap_3state.ipynb`の下記のパスを変更して上から順に実行してください．
            - 以下の例だと，何もしていない`Cop1_KO`，`Cop1_KO`状態のものに対して，遺伝子`Apoe`を作成したもの，`Cop1_WT`の3状態でUMAPを可視化するような設定になっています．
            
            ```python
            # ----- 入力 Arrow -----
            ARROW_PATH = "/path/to/data.arrow"  # UMAPを可視化したいarrowデータのパスを指定
            # /work/scRNA-seq_data/Adrenal_scRNA-seq_MS/output_arrow/data-00000-of-00001.arrow
            # /work/eval_dataset/Cop1KO_isp_mouse_tokenize_dataset/data-00000-of-00001.arrow
            CONDITION_COL = "disease"
            
            # ==== Cell 3: パラメータの追加/変更 ====
            VALUE_KO = "Cop1_KO" # 遺伝子削除を行う方のdisease名を指定
            VALUE_WT = "Cop1_WT" # 遺伝子削除後に近づけたい目標状態を指定
            
            # シナリオ（削除する遺伝子集合）—例
            MULTI_DELETE_GENE_SETS = [
                ["Apoe"], 
                # ["Apoe", "Lrp1"],
                # ["Apoe", "Lrp1", "Ldlr"],
            ]
            
            # ----- 辞書 -----
            PATH_GENE_SYMBOL2ENS = "/path/to/MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
            TOKEN_DICTIONARY_FILE = "/path/to/MLM-re_token_dictionary_v1.pkl"
            
            # ----- 作業/出力 -----
            WORK_DIR = Path("/path/to/out/folder") # UMAPを出力したいフォルダのパスを指定
            ...
            FINETUNED_MODEL_DIR = "/path/to/fine-tuning/model"  # ←あなたのモデルパス
            ```