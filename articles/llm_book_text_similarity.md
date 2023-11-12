---
title: "VRAMが少ない環境でLLMを効率的にfine-tuneしてベクトル検索を実現する"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["LLM", "ANN", "Python", "ベクトル検索"]
published: false
---

LLM周りの基本的な知識とTransformersをもっと扱えるようになりたくて、最近 [大規模言語モデル入門](https://www.amazon.co.jp/dp/B0C9P7K6VH)を読んでいたのですが、その中で「メモリ効率の良いファインチューニング」という節が面白くて色々自分で試してみていたりしました。ここでは、自分の手元で文章の類似度を計算するモデルをファインチューンして作って見たので、それについて書きたいと思います。

## 実験環境

- Ubuntu 20.04
- NVIDIA RTX2080 (VRAM: 8GB)
- Python 3.11

## 実験

文章の類似度を計算するモデルを作るために、JGLUEのJSTSというデータセットを利用しました。このデータセットはHugging Face上から取得することが可能で、以下のようなカラムを持ったデータを使うことが可能です。

https://huggingface.co/datasets/llm-book/JGLUE

- `sentence1`: 1つめの文章
- `sentence2`: 2つめの文章
- `label`: 文章間の類似度(0 ~ 5)

1つ1つの文章の長さは短めで、長くても80文字程度になっています。そしてファインチューンのベースとしたモデルはLINEヤフーが出している `line-distilbert-base-japanese` というモデルを利用しました。

### ベース実装

```python
class FineTuneModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.loss_fn = nn.MSELoss()

        self.global_step = 0

    def get_normlaized_embedding(self, sentence: BatchEncoding) -> Tensor:
        embedding = self.text_encoder(**sentence).last_hidden_state.mean(dim=1)
        return embedding / torch.norm(embedding, p=2, dim=-1, keepdim=True)

    def forward(
        self, sentence1: BatchEncoding, sentence2: BatchEncoding, similarities: Tensor
    ) -> ModelOutput:
        sentence1_embedding = self.get_normlaized_embedding(sentence1)
        sentence2_embedding = self.get_normlaized_embedding(sentence2)
        predicted_similarities = (sentence1_embedding * sentence2_embedding).sum(dim=1)
        loss = self.loss_fn(predicted_similarities, similarities)

        if self.training:
            self.global_step += 1

        return ModelOutput(loss=loss, logits=predicted_similarities)
```

類似度を学習させるために `FineTuneModel` というモデルでベースモデルをラップしています。このモデルの構造自体はとてもシンプルで、ベースモデルでembeddingを取得した後にノーマライズしているだけです。学習時の損失関数は、2つのembeddingどうしのコサイン類似度とデータセットで与えられている類似度の間の二乗誤差になっています。

### 勾配チェックポインティングを使ったモデル

transformersのモデルを単純にファインチューンする場合には `TrainingArguments` の中で `gradient_checkpointings=True` と指定してやればいいだけなのですが、今回は変にモデルをラップしてしまっているせいでそれができません。しかしTransformersの `PretrainedModel` 内の実装を見てみると `gradient_checkpointing_enable()` を使ってON/OFFを切り替えているだけであることがわかったので、明示的にこのメソッドを呼び出すだけで実現できました。

https://github.com/huggingface/transformers/blob/7ee995fd9c692761c4601ddbffa2ac2ec9f27b0b/src/transformers/modeling_utils.py#L1161-L1165

なのでベース実装との変更点はほどんどなく以下のように実装が可能です。

```python
class FineTuneModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.text_encoder.gradient_checkpointing_enable()  # <- 追加
        self.loss_fn = nn.MSELoss()

        self.global_step = 0

    # ここから下は同じ
```

### 学習実行用のスクリプト

以上のモデルを学習させるためのコードは以下のようになります。

```python
def collelate_fn(
    examples: list[dict[str, str | float]], tokenizer: AutoTokenizer
) -> dict[str, Any]:
    sentence1: str = []
    sentence2: str = []
    similarities: list[float] = []

    for example in examples:
        sentence1.append(example["sentence1"])
        sentence2.append(example["sentence2"])
        similarities.append(example["label"])

    tokenized_sentence1 = tokenizer(
        sentence1, max_length=512, return_tensors="pt", padding=True, truncation=True
    )
    tokenized_sentence2 = tokenizer(
        sentence2, max_length=512, return_tensors="pt", padding=True, truncation=True
    )

    del tokenized_sentence1["token_type_ids"]
    del tokenized_sentence2["token_type_ids"]

    return {
        "sentence1": tokenized_sentence1,
        "sentence2": tokenized_sentence2,
        "similarities": torch.tensor(similarities) / 2.5 - 1,
    }


if __name__ == "__main__":
    model_name = "line-corporation/line-distilbert-base-japanese"
    train_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="train")
    valid_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="validation")
    model = FineTuneModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    batch_size = 128
    output_dir = f"./output_jsts_baseline_{batch_size}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        num_train_epochs=30,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        label_names=["similarities"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=lambda x: collelate_fn(x, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

    model.text_encoder.save_pretrained(output_dir)

```

### 実験結果

この2つの実装を比較してみます。勾配チェックポインティングは訓練結果に影響を与えるものではなく、メモリ使用量を抑える代わりに計算スピードが少し落ちるといった類のものです。なので比較対象としては

- メモリ使用量
- 計算スピード

の2点を見ます。結果をまとめると以下のようになります。

モデル | batch size | 計算速度(iteration / sec) | VRAM使用量
-- | -- | -- | --
baseline | 128 | 4.3 | 7.8 GB
baseline | 256 | OOM | >8.0 GB
勾配チェックポインティング | 128 | 4.0 | 3.6 GB
勾配チェックポインティング | 512 | 1.0 | 5.8 GB

計算速度は確かに 4.3 → 4.0へと7 %ほど低下していますが大した変化ではないように見えます。しかしVRAMの使用量は半分以下まで激減しており、batch sizeを4倍の512まで上げてもOOMはしませんでした。

## 類似文書のベクトル検索

文章の類似度を学習させたモデルを作ったので、これを使って類似文書の検索を試してみます。ベクトル検索を行うためのライブラリは様々ありますが、ここでは最近SpotifyがOSSとして公開したVoyagerを試してみたいと思います。

https://spotify.github.io/voyager/

```python
import pickle

from datasets import load_dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from voyager import Index, Space


def calc_embedding(model, tokenizer, texts) -> np.ndarray:
    input = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    del input["token_type_ids"]

    with torch.no_grad():
        output = model(**input)

    embedding = output.last_hidden_state.mean(dim=1)

    return embedding[0].numpy()


if __name__ == "__main__":
    model = AutoModel.from_pretrained(
        "./output_jsts_grad_ckpt_512", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "line-corporation/line-distilbert-base-japanese"
    )

    create_index = True
    if create_index:
        """インデックスを作成する場合"""
        valid_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="validation")
        texts = []
        for example in valid_dataset:
            texts.append(example["sentence1"])
            texts.append(example["sentence2"])

        texts = list(set(texts))
        with open("./texts_list", "wb") as f:
            pickle.dump(texts, f)

        index = Index(Space.Cosine, num_dimensions=768)
        batch_size = 128
        for i in tqdm(range(0, len(texts) // batch_size + 1)):
            embeddings = texts[i * batch_size : (i + 1) * batch_size]
            for embedding in embeddings:
                _ = index.add_item(calc_embedding(model, tokenizer, embedding))

        index.save("index.voy")
    else:
        """作成済みインデックスを使う場合"""
        index = Index.load("index.voy")
        texts = pickle.load(open("./texts_list", "rb"))

    neighbors, distances = index.query(
        calc_embedding(model, tokenizer, "ジャンプ台から男性スケートボーダーがジャンプしています。"), k=2
    )

    print(texts[neighbors[0]])
    print(texts[neighbors[1]])

```

上のコードで試している例はvalidationデータ内にある文章です。これに近い文章を引っ張ってきた結果は以下のようになりました。

```bash
ジャンプ台から男性スケートボーダーがジャンプしています。
スケートボーダーがジャンプ台でジャンプしています。
```

インデックス内に存在する文章で検索したのでもちろんtopはクエリの文書それ自体が返ってきます。2番目に近い文章もスケートやジャンプといった共通点があるので近い文章と言えそうです。次に、自分で考えた文章を適当に入れてみます。「今日はずっと家で本を読んでいました。」という文章を入れた結果が以下です。

```bash
キッチンで何もせずに話している
リビングで飲みものを飲む人たち
```

「ずっと家で」という部分がのんびりしているようなニュアンスを与えるのか、家の中でのんびりしている感じの文章が近いと判定されています。

## 感想

Transformersは今まで軽いファインチューンに使ったことはあったのですが、色々工夫してみるといった部分はしたことがなかったのでとても勉強になりました。本当はLoRAを試すということもやってみたかったのですが、今回はうまく動かせず断念しました。今度再チャレンジしてみようと思います。

最後のベクトル検索の部分は本の内容からは少し離れて、自分が最近気になっていたライブラリを試すということをやってみました。Voyagerはとてもシンプルでドキュメントもしっかりしているということは聞いていて知ってはいたのですが、自分で実際に試してみてやりたいことをすぐに実現できたので体験はものすごく良かったです。Voyagerの性能に関しては確認したことがなかったので今度試してみたいです。