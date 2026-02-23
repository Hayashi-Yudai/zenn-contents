---
title: "NVIDIA-Nemotron-Nano-9B-v2-Japanese から Embedding モデルを作る"
emoji: "🔥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["LLM", "Embedding", "NVIDIA-Nemotron-Nano-9B-v2-Japanese"]
published: false
---

NVIDIA が公開した [NVIDIA-Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) は、Qwen3 の 3〜6 倍ものスループットを持つ日本語 LLM として注目を集めています。ただ、現状では Embedding モデルは提供されていません。

私は個人的に記事推薦のモデルを作って運用していて、自然言語を Embedding に変換して機械学習モデルの特徴量に入れるということをしています。もし Nemotron ベースの高速な Embedding モデルがあれば、より速く性能の高い記事推薦ができるのではないか——ということで、自分で作ってみることにしました。

実験コードはこちらのリポジトリにまとめています。

https://github.com/Hayashi-Yudai/Nemotron-nano-embedding-train

## どうやって Embedding モデルを作るか


今回のアプローチはシンプルで、LLM の最終層から得られる hidden_state を Mean Pooling で集約して固定長のベクトルにし、それを文全体の Embedding として使います。
ただし、LLM をそのまま使っただけでは Embedding としてうまく機能しません。LLM は「次のトークンを予測する」という目的で学習されているため、各トークンの hidden_state は直前までの文脈から次に何が来るかを表現するように最適化されています。文全体の意味を一つのベクトルにまとめるという用途には合っていないわけです。実際、学習前のモデルで類似度を計算してみると、意味が全く異なる文でも高い類似度が出てしまうことがあります。

そこで、意味が近い文どうしのベクトルが近く、遠い文どうしは離れるように対照学習で追加学習を行います。

## 実験設計

### ベースライン

比較対象には Qwen3-Embedding Family (0.6B, 4B, 8B) を使いました。日本語対応の Embedding モデルとしては現状トップクラスの性能を持つモデルです。9B パラメータの汎用 LLM から作った Embedding モデルが、専用に学習された Embedding モデルとどこまで戦えるかを見てみます。

### データセット

学習・評価には以下の 2 つのデータセットを使いました。

- **mMARCO Japanese**: 大規模な情報検索データセット。クエリと関連パッセージのペアが含まれており、ここから 50,000 件をサンプリングして使いました。検索タスクに必要な「クエリと関連文書を近づける」能力をまず身につけさせるのが狙いです。
- **JSTS (Japanese Semantic Textual Similarity)**: 日本語の文ペアに 0〜5 の類似度スコアが付いたデータセットです。文の意味的な近さを細かく捉える能力を学習するために使います。評価もこのデータセットの validation split で行いました。

### 学習手法

- **LoRA (r=16, alpha=32)**: 9B のモデルをフル学習するのは現実的ではないので、LoRA でパラメータ効率よく学習します。attention と FFN の projection 層 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) にアダプタを挿入しています。
- **Contrastive Learning (In-batch Negatives)**: 同じバッチ内の他のペアをネガティブサンプルとして使う対照学習です。バッチ内の文 A と文 B のコサイン類似度行列を計算し、対角要素（正しいペア）のスコアが高くなるように学習します。バッチサイズが大きいほどネガティブサンプルが増えて効果的なので、今回は 256 で学習しました。

学習は 2 段階で行いました。まず mMARCO で大まかな検索能力を学習し、その後 JSTS で意味類似度を仕上げるという流れです。

## 結果

JSTS に用意されている評価用データセット (1,457 件) を使って評価しました。各文ペアの Embedding 間のコサイン類似度を計算し、それを 0〜5 のスコアにマッピングして正解ラベルと比較しています。

Nemotron については、JSTS だけで学習したモデルと、mMARCO → JSTS の 2 段階で学習したモデルの両方を載せています。2 段階学習の効果がどれくらいあるかを見るためです。

| モデル                  | 学習              |  Spearman |   Pearson |       MAE |
| ----------------------- | ----------------- | --------: | --------: | --------: |
| Nemotron-Nano-9B (ours) | JSTS のみ         |     0.813 |     0.869 |     2.084 |
| Nemotron-Nano-9B (ours) | mMARCO 50k → JSTS | **0.832** | **0.882** |     2.098 |
| Qwen3-Embedding-0.6B    | 事前学習済み      |     0.807 |     0.854 | **1.835** |
| Qwen3-Embedding-4B      | 事前学習済み      |     0.829 |     0.872 |     1.866 |
| Qwen3-Embedding-8B      | 事前学習済み      |     0.837 |     0.879 |     1.953 |

いくつかわかったことがあります。

- **2 段階学習の効果が大きい**: JSTS だけで学習したモデルと比べて、mMARCO で事前学習してから JSTS で仕上げたモデルは Spearman が 0.813 → 0.832 と大きく改善しました。汎用的な検索能力をまず身につけてから意味類似度を学習するという順番が効いているようです。
- **Qwen3-Embedding-8B に迫る性能**: 専用の Embedding モデルである Qwen3-Embedding-8B の Spearman 0.837 に対して 0.832 とほぼ同等の水準まで来ています。Pearson では 0.882 と上回りました。
- **MAE は高め**: 一方で MAE（予測スコアと正解の平均絶対誤差）は Qwen3 系より大きくなっています。順位相関は高いものの、スコアの絶対値の予測精度には改善の余地がありそうです。

