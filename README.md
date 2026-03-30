# LLM Visualizer

**ニューラルネットワーク アーキテクチャ ビジュアライザー** — Python ソースファイル (.py)、HuggingFace config.json、GGUF ファイルからモデル構造を読み取り、左→右のフローで可視化する Windows デスクトップアプリケーションです。

<p align="center">
  <img src="docs/screenshot_placeholder.png" alt="LLM Visualizer Screenshot" width="800">
  <br>
  <em>※ アプリ実行後のスクリーンショットに差し替えてください</em>
</p>

---

## ✨ 主な機能

| 機能 | 説明 |
|------|------|
| **マルチフレームワーク対応** | PyTorch, TensorFlow/Keras, JAX/Flax, JAX/Haiku, PaddlePaddle, MindSpore |
| **複数ファイル形式** | Python (.py), JSON (config.json), GGUF (.gguf) |
| **自動フレームワーク検出** | import 文から使用フレームワークを自動判定 |
| **レイヤー可視化** | 左→右フローのアーキテクチャ図を GDI+ で描画 |
| **コンテナ展開** | ModuleList / Sequential / ModuleDict の子レイヤーを展開表示 |
| **サブクラス展開** | ユーザー定義サブモジュールクラスを再帰的に展開 |
| **メタデータ抽出** | VocabSize, HiddenSize, NumAttentionHeads 等を自動推定 |
| **ズーム & パン** | マウスホイールでズーム、ドラッグでパン |
| **PNG エクスポート** | アーキテクチャ図を高解像度 PNG として保存 |
| **ドラッグ＆ドロップ** | ファイルをウィンドウにドロップして即座に読み込み |

---

## 🏗️ 対応フレームワーク詳細

### PyTorch
- `nn.Module` サブクラスの `__init__()` / `forward()` を解析
- `nn.ModuleList`, `nn.ModuleDict`, `nn.Sequential` (+ `OrderedDict`) 対応
- `nn.TransformerEncoder` / `nn.TransformerDecoder` の内部構造展開
- `F.relu`, `torch.relu` 等の関数型アクティベーション検出

### TensorFlow / Keras
- Sequential API (`model.add()` / インライン定義)
- Functional API (`layers.Dense(...)(x)`)
- `activation='relu'` 等のアクティベーション引数抽出

### JAX / Flax
- `setup()` メソッドによるレイヤー定義
- `@nn.compact` デコレータによるインライン定義 (`nn.Dense(features=128)(x)`)
- Flax 固有型の正規化 (`Embed` → `Embedding`, `SelfAttention` → `MultiheadAttention`)
- キーワード引数による次元抽出 (`features=`, `num_heads=`, `num_embeddings=`)

### JAX / Haiku
- `hk.Module` サブクラスの `__init__()` / `__call__()` を解析
- `hk.Linear`, `hk.Embed`, `hk.LayerNorm` 等のレイヤー認識

### PaddlePaddle
- `nn.Layer` サブクラスの `__init__()` / `forward()` を解析
- `nn.LayerList` コンテナ対応
- `paddle.nn.` プレフィックスのレイヤー認識

### MindSpore
- `nn.Cell` サブクラスの `__init__()` / `construct()` を解析
- `nn.CellList` コンテナ対応
- `mindspore.nn.` / `ms.nn.` プレフィックスのレイヤー認識

---

## 🚀 動作要件

| 要件 | バージョン |
|------|-----------|
| **OS** | Windows 10 / 11 (x64) |
| **.NET** | .NET 10 Preview 以上 |

> [!NOTE]
> .NET 10 ランタイムがインストールされていない場合は [公式ダウンロードページ](https://dotnet.microsoft.com/download/dotnet/10.0) から入手してください。

---

## 📦 インストール & 実行

### リリースバイナリを使用する場合

1. [Releases](https://github.com/freyWylfred/LLMVisualizer/releases) ページから最新の ZIP をダウンロード
2. 任意のフォルダに展開
3. `LLMVisualizer.exe` を実行

### ソースからビルドする場合

```bash
git clone https://github.com/freyWylfred/LLMVisualizer.git
cd LLMVisualizer
dotnet build
dotnet run --project LLMVisualizer
```

---

## 🧪 テスト

50 件のユニットテスト（MSTest）で主要な機能を検証しています。

```bash
dotnet test
```

### テストカバレッジ

| カテゴリ | テスト数 | 内容 |
|---------|---------|------|
| PyTorch | 14 | GPT, Transformer, CNN, Sequential, LSTM |
| TensorFlow | 2 | Sequential, Activation |
| JAX/Flax | 8 | setup(), @nn.compact, Embed, Dense, SelfAttention |
| JAX/Haiku | 5 | Parse, Embed, Linear, Framework, CallOrder |
| PaddlePaddle | 5 | Parse, Embed, Framework, Forward, SubClass |
| MindSpore | 6 | Parse, Embed, Framework, Construct, SubClass, Metadata |
| エッジケース | 4 | 空ファイル, モジュールなし, 不正括弧, ファイル不在 |
| JSON/GGUF | 4 | 不在ファイル, 不正JSON, 不正マジック, 切詰ファイル |
| ユーティリティ | 2 | FormatParameters |

---

## 📁 プロジェクト構成

```
LLMVisualizer/
├── LLMVisualizer/                # メインアプリケーション
│   ├── Program.cs                # エントリーポイント
│   ├── Form1.cs                  # メインフォーム (ファイル読込, D&D, UI)
│   ├── Form1.Designer.cs         # フォームデザイナー
│   ├── ArchitecturePanel.cs      # GDI+ カスタム描画パネル
│   ├── LLMModel2.cs              # モデル/パーサー (Python, JSON, GGUF)
│   ├── LLMVisualizer.csproj      # プロジェクトファイル
│   └── TestData/                 # テスト用 Python サンプル
│       ├── test_pytorch_basic.py
│       ├── test_pytorch_transformer.py
│       ├── test_pytorch_cnn.py
│       ├── test_pytorch_sequential.py
│       ├── test_pytorch_lstm.py
│       ├── test_tf_sequential.py
│       ├── test_flax_setup.py
│       ├── test_flax_compact.py
│       ├── test_haiku.py
│       ├── test_paddle.py
│       ├── test_mindspore.py
│       ├── test_empty.py
│       ├── test_no_module.py
│       └── test_malformed.py
├── LLMVisualizer.Tests/          # ユニットテスト (MSTest)
│   ├── ParserTests.cs            # 50 テストケース
│   ├── LLMVisualizer.Tests.csproj
│   └── TestData/                 # テストデータコピー
├── LLMVisualizer.slnx            # ソリューションファイル
├── LICENSE                       # MIT License
└── README.md
```

---

## 🎨 レイヤーの色分け

| 色 | レイヤータイプ |
|----|---------------|
| 🔵 青 | Embedding, Embed, Input |
| 🟣 紫 | LayerNorm, RMSNorm, BatchNorm |
| 🟢 緑 | MultiheadAttention, Transformer, LSTM, GRU |
| 🟠 オレンジ | Linear, Dense, DenseGeneral |
| 🔴 赤 | Output / Head レイヤー |
| 🟡 黄 | Conv, Pool |
| 灰緑 | ReLU, GELU, Dropout 等のアクティベーション |
| ⬜ 灰 | その他 |

---

## 📝 使い方

1. **ファイルを開く**: メニュー「ファイル → 開く」または、ウィンドウにファイルをドラッグ＆ドロップ
2. **対応形式**:
   - `.py` — Python ソースファイル（PyTorch, TensorFlow, Flax, Haiku, Paddle, MindSpore）
   - `.json` — HuggingFace config.json
   - `.gguf` — GGUF モデルファイル
3. **操作**:
   - **ズーム**: マウスホイール or メニュー「表示 → ズームイン / ズームアウト」
   - **パン**: マウスドラッグ
   - **リセット**: メニュー「表示 → ズームリセット」
4. **エクスポート**: メニュー「ファイル → PNG エクスポート」で高解像度画像を保存

---

## 🤝 コントリビューション

Issue や Pull Request は歓迎します。

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Request を作成

---

## 📜 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

---

## 🙏 謝辞

- [.NET 10](https://dotnet.microsoft.com/) — Microsoft
- [Windows Forms](https://learn.microsoft.com/dotnet/desktop/winforms/) — GDI+ ベースの UI フレームワーク
