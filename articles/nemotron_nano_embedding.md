---
title: "NVIDIA-Nemotron-Nano-9B-v2-Japanese から Embedding モデルを作る"
emoji: "🔥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["LLM", "Embedding", "NVIDIA-Nemotron-Nano-9B-v2-Japanese"]
published: true
---

NVIDIA が公開した [NVIDIA-Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) は、Qwen3 の 3〜6 倍ものスループットを持つ日本語 LLM として注目を集めています。現状ではチャット形式での利用のためのモデルのみが公開されており Embedding モデルは提供されていないという状況になっています。

私は個人的に記事推薦のモデルを作って運用していて、自然言語を Embedding に変換して機械学習モデルの特徴量として使っています。Nemotron ベースの高速な Embedding モデルがあれば、推論速度と推薦性能の両方を改善できるのではないか——ということで、自分で作ってみることにしました。この記事では、Embedding モデルの学習とパブリックデータセットでの評価の部分までを書きます。

実験コードは以下のリポジトリにまとめています。

https://github.com/Hayashi-Yudai/Nemotron-nano-embedding-train

## どうやって Embedding モデルを作るか

以下のような手順でモデルの学習を行いました。この学習手順自体は一般的によく知られている手法だと思っています。

**1. LLM に文を入力し、最終層の hidden_state を取得する**

LM ヘッドは使わず、backbone だけを通して hidden_state を取り出します。

```python
backbone = get_backbone_model(model)  # LM ヘッドを除いた部分
outputs = backbone(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
)
```

**2. Mean Pooling で固定長のベクトル（= Embedding）にする**

各トークンの hidden_state を attention_mask で重み付けして平均を取ります。

```python
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
```

**3. 対照学習で追加学習する**

LLM は「次のトークンを予測する」ために学習されているので、hidden_state をそのままプーリングしても文の意味はうまく表現できません。学習前のモデルで試すと、意味が全く異なる文でも高い類似度が出てしまいます。

そこで、文ペア (A, B) のバッチに対してコサイン類似度の行列を作り、正しいペア（対角要素）のスコアが最大になるように学習します。

```python
a_emb = F.normalize(encode_texts(model, tokenizer, text_a, cfg), dim=-1)
b_emb = F.normalize(encode_texts(model, tokenizer, text_b, cfg), dim=-1)

logits = (a_emb @ b_emb.T) / cfg.temperature
labels = torch.arange(logits.size(0), device=logits.device)

loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
```

データ内のペア(A, B)のみがポジティブサンプル、バッチ内の他のテキストは全てネガティブサンプルという扱いになります。

## 実験設計

### ベースライン

比較対象には Qwen3-Embedding (0.6B, 4B, 8B) を使いました。日本語対応の Embedding モデルで、最近よく利用されているのを目にすることが多いモデルです。汎用 LLM から作った Embedding モデルが、専用に学習されたモデルとどこまで戦えるかを見てみます。

### データセット

学習・評価には以下の 2 つのデータセットを使いました。

- **mMARCO Japanese**: 大規模な情報検索データセットで、クエリと関連パッセージのペアが含まれています。ここから 50,000 件をサンプリングして使いました。「クエリと関連文書を近づける」という汎用的な検索能力をまず身につけさせるのが狙いです。
- **JSTS (Japanese Semantic Textual Similarity)**: 日本語の文ペアに 0〜5 の類似度スコアが付いたデータセットです。文の意味的な近さを細かく捉える能力の学習と、評価の両方に使いました。

### 学習手法

- **LoRA (r=16, alpha=32)**: LoRAを利用して学習を行いました。attention と FFN の projection 層 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) にアダプタを挿入しています。
- **Contrastive Learning (In-batch Negatives)**: 同じバッチ内の他のペアをネガティブサンプルとして使う対照学習です。バッチ内の文 A と文 B のコサイン類似度行列を計算し、対角要素（正しいペア）のスコアが高くなるように学習します。mMARCOの学習時には32, JSTSの学習時には256を利用しました。これは単純に使っているGPUで利用できる最大長にしたという設定です。

学習は 2 段階で行いました。まず mMARCO で大まかな検索能力を学習し、その後 JSTS で意味類似度を仕上げるという流れです。

### 学習環境

- GPU: A100 (80 GB) x 1
- RAM: 170 GB
- Python 3.12

## 結果

JSTS の validation split (1,457 件) で評価しました。各文ペアの Embedding 間のコサイン類似度を 0〜5 のスコアにマッピングし、正解ラベルと比較しています。

Nemotron については、2 段階学習の効果を確認するために、JSTS のみで学習したモデルと mMARCO → JSTS の 2 段階で学習したモデルの両方を載せています。

| モデル                  | 学習              |  Spearman |   Pearson |       MAE |
| ----------------------- | ----------------- | --------: | --------: | --------: |
| Nemotron-Nano-9B (今回学習したもの)| JSTS のみ         |     0.813 |     0.869 |     2.084 |
| Nemotron-Nano-9B (今回学習したもの) | mMARCO 50k → JSTS | **0.832** | **0.882** |     2.098 |
| Qwen3-Embedding-0.6B    | 事前学習済み      |     0.807 |     0.854 | **1.835** |
| Qwen3-Embedding-4B      | 事前学習済み      |     0.829 |     0.872 |     1.866 |
| Qwen3-Embedding-8B      | 事前学習済み      |     0.837 |     0.879 |     1.953 |

結果からいくつかのことがわかりました。

- **2 段階学習の効果が大きい**: mMARCO で事前学習してから JSTS で仕上げたモデルは、JSTS だけで学習したモデルと比べて Spearman が 0.813 → 0.832 と大きく改善しました。汎用的な検索能力をまず身につけてから意味類似度を学習する、という順番が効いているようです。
- **Qwen3-Embedding-8B に迫る性能**: Qwen3-Embedding-8B の Spearman 0.837 に対して 0.832 と、ほぼ同等の水準です。Pearson では 0.882 とこちらが上回りました。
- **MAE は高め**: 一方で MAE（予測スコアと正解の平均絶対誤差）は Qwen3 系より大きくなっています。順位の相関は高いものの、スコアの絶対値についてはまだ改善の余地がありそうです。

## まとめ

NVIDIA-Nemotron-Nano-9B-v2-Japanese を追加学習して Embedding 計算に利用できるようにしてみました。使っているデータも少量で、学習もシンプルなことしかしていませんが Qwen3-Embedding-8B と同等程度の性能が出ることがわかりました。Qwen の方は別に日本語特化のモデルというわけではないので、ちゃんとやれば NVIDIA-Nemotron-Nano-9B-v2-Japanese の方は日本語における性能は高くなるのではないかと考えています。技術記事の推薦の特徴量生成に使えないか、という点をモチベーションに実験していたので、今後は技術系のワードに対して性能がちゃんと出るのか、英語が混ざっているテキストに対しても性能が出るのか、という点の評価はしていきたいと思っています。
