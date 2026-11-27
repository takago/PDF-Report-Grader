# チュートリアル：サンプルレポートを採点する

このチュートリアルでは，架空の組込みシステム演習の提出物を使い，名簿照合，フォーム回答抽出，画像認識，AI採点，Excel集計，採点済PDFの生成を確認します．実際の学生情報は含まれていません．回路写真とIDE画面は模擬画像であり，この演習では添付ありとして扱います．

## 1．準備

Linux/macOSのシェルで，プロジェクトのルートディレクトリから以下を実行します（Python 3.11以降）．既にセットアップ済みなら省略できます．

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp -n examples/config.example.toml config.toml
```

OpenAI互換APIを提供するLLMと画像入力対応VLMを用意し，`config.toml` の既存セクション内の対応項目を編集します．以下は抜粋です．URL，モデル名，APIキーは各自の環境に合わせてください．

```toml
[output]
directory = "out"

[vlm]
enabled = true
base_url = "http://localhost:8000/v1"
model = "YOUR_VLM_MODEL"
api_key = ""

[grading]
base_url = "http://localhost:8000/v1"
model = "YOUR_LLM_MODEL"
api_key = ""
visual_mode = "auto"
annotate_pdf = true
```

同じセクションを重複して追加しないでください．APIキーは環境変数 `VLM_API_KEY`，`GRADING_API_KEY` でも指定できます．既存の設定ファイルは上書きしないでください．モデルが `reasoning_effort` を受け付けない場合は，対応する設定を `"auto"` にします．

## 2．サンプルを配置する

配布データは `examples/demo/in/` に保存してあります．実データを上書きしないよう，新規のプロジェクト環境で実行してください．

```bash
mkdir -p in
cp -rn examples/demo/in/. in/
```

同名ファイルが既にある場合はコピーされないため，既存の課題と混在させず，別の作業用コピーを使用してください．

```text
in/
├── kadai_w01.pdf          # A4 2ページ，フォーム付き雛形
├── grading_w01.txt        # 自然言語の採点基準
├── DEMO.xlsx              # 架空学生11人の名簿
└── DEMO/
    └── DATA_w01_demo.zip  # 10人分の提出PDF
```

ZIPにはフォーム保持8件，平坦化1件，全ページ画像化1件が含まれます．1人は意図的に未提出です．元のサンプルと採点基準は変更していません．

## 3．実行する

### WebUI

```bash
.venv/bin/python webui.py
```

`http://127.0.0.1:7860/` を開き，DEMO / w01（DATA_w01_demo）を選択します．AI採点を有効にし，必要に応じて「採点プラン確認・生成」で採点プランを確認してから処理を開始します．ログとPDFプレビューで進行を確認できます．WebUIはローカル専用です．

### CLI（WebUIの代わりに実行）

```bash
.venv/bin/python grade_reports.py \
    in/kadai_w01.pdf \
    in/DEMO/DATA_w01_demo.zip \
    in/DEMO.xlsx \
    --grading in/grading_w01.txt \
    --output-dir out/DEMO
```

両方から同時に同じ課題を実行しないでください．初回は採点プランを生成します．再実行時には有効なプランやキャッシュを再利用します．

## 4．結果を確認する

`out/DEMO/DATA_w01_demo/` の `DATA_w01_demo.xlsx` を開きます．`Results` シートの学籍番号，`status`，`AI点数`，`AI採点理由`，`AI採点状態` を確認してください．`graded/` には採点理由付きPDFが生成されます．`grading-plan.json` は生成された採点プランです．未提出や未照合の確認には `Unmatched` シートも利用してください．

| 学籍番号 | 確認項目 | 期待点 |
|---|---|---:|
| 9000001 | 全課題正解 | 10 |
| 9000002 | 単位換算・フォーム平坦化 | 10 |
| 9000003 | 時間の誤答 | 9 |
| 9000004 | 電圧の許容範囲外 | 9 |
| 9000005 | static変数の誤説明 | 8 |
| 9000006 | 波形の周期の誤り | 9 |
| 9000007 | デューティ比の誤り | 9 |
| 9000008 | 回路写真・スクリーンショットなし | 8 |
| 9000009 | 考察不足・全ページ画像化 | 9 |
| 9000010 | 全課題未記入（提出者最低点） | 1 |
| 9000011 | 未提出 | 0 |

機械可読な期待点は [expected.csv](expected.csv) にあります．期待点は採点基準から算出した設計値であり，AIの実測結果ではありません．特に波形認識・意味的な判定はモデルによって異なります．REVIEWの暫定点や期待点との不一致は教員が確認してください．未提出者も照合漏れを確認するためREVIEWとなります．

全課題未記入の学生9000010では，空欄の回答を根拠として扱い，正しく減点されることを確認します．波形未記入の場合は周期・デューティ比の減点を重複させません．

## 5．再実行・トラブルシューティング

- API接続やモデル名を確認し，VLMが画像入力に対応していることを確認してください．
- `[logging]` の `level = "debug"` を設定すると応答検証エラーの詳細を確認できます．デバッグログには回答内容が含まれるため，実データでの共有には注意してください．
- 同じ入力の再採点では，展開済みPDF・認識キャッシュ・採点プランが再利用されます．新規試験には別の作業用コピーを使ってください．既存XLSXは再実行時に退避されます．
- APIなしでプログラムの単体テストを実行するには，ルートディレクトリで `.venv/bin/python -m pip install -r requirements-dev.txt` の後に `.venv/bin/python -m unittest discover -s tests` を実行します．モックによる回帰テストは実モデルの採点精度を保証しません．

詳細は [プロジェクトREADME](../../README.md) を参照してください．
