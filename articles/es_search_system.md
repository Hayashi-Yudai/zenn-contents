---
title: "ElasticsearchとXGBoostを組み合わせた検索ランキング作成と評価"
emoji: "🤖"
type: "tech"
topics: ["Elasticsearch", "検索", "機械学習"]
published: false
---

最近検索周りで「[機械学習による検索ランキング改善ガイド](www.amazon.co.jp/dp/4814400306)」という本が出版されて気になって読んでみたので、それを読んで勉強しつつ手を動かしてみてわかったことや感想を紹介してみようと思います。

## この記事に書くこと＆書かないこと

この記事では以下のようなことに焦点を当てて書きます。

- ElasticsearchとXGBoostを組み合わせたときの性能・負荷変化の実験
- Elasticsearch上での特徴量エンジニアリングの体験

逆に、実験の下準備といったことについては最低限しか書かないので具体的なElasticsearchの使い方等については他の記事もしくは書籍を参照してください。

## 実験を行う検索システムの構成

書籍で使われているコードをベースとして色々と自分で実験を行いました。

https://github.com/oreilly-japan/building-search-app-w-ml/tree/master


## 実験を通して知りたかったこと

自分の手を動かして実験することで知りたかったことをまとめると以下のようになります。

- Elasticsearchについて知りたい
    - そもそもどうやって使うのか
    - LTRプラグインの使い方と開発体験
- 機械学習モデルを活用した検索ランキングについて知りたい
    - どうやってElasticsearchに機械学習モデルを組み込むのか
    - 生のElasticsearchよりどれくらい性能・負荷が変化するのか


## 実験環境

- Dataset: simplewiki-202109-pages-with-pageviews-20211001
- Elasticsearch v7.13.4
- Python 3.11.6
- 実験に使ったマシンのスペック
    - OS: Ubuntu 22.04
    - CPU: Intel Core i7-9700K
    - RAM: DDR4 64GB

## 下準備

### Elasticsearchへのリクエスト

Elasticsearchへリクエストを送る際にはjson形式でのリクエストを送ります。例えばElasticsearchにアップロードしたsimplewikiから本文(`text`)中に "ninja" という文字列を含む記事のタイトルを10件取得するクエリは以下のようになります。

```json
GET /simplewiki/_search
{
  "query": {
    "match": {
      "text": "ninja"
    }
  },
  "size": 10,
  "_source": ["title"]
}
```

KibanaのDev Toolから上のリクエストを送ると以下のような出力が返ってきます。

```json
{
  "took" : 11,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 17,
      "relation" : "eq"
    },
    "max_score" : 14.577337,
    "hits" : [
      {
        "_index" : "simplewiki",
        "_type" : "_doc",
        "_id" : "440219",
        "_score" : 14.577337,
        "_source" : {
          "title" : "TMNT (movie)"
        }
      },
      // ...
    ]
  }
}
```

`took` はリクエストにかかった時間で、`hits.hits` 以下にはレスポンスについての情報が格納されています。`_score` はElasticsearchで計算された記事のスコアで、デフォルトではBM25の値になります。

### featuresetへの特徴量情報のアップロード

XGBoostと組み合わせるために、モデルの学習に使う特徴量の情報をElasticsearch上に登録します。これには、LTR (Learning to Rank) プラグインを使います。

https://elasticsearch-learning-to-rank.readthedocs.io/en/latest

json形式で登録したい特徴量の情報を用意してそれをリクエストと一緒に送ることでElasticsearchに特徴量を登録します。`match_explorer` という機能を使えば基本的な統計量が計算できます。書籍ではこれをフルに使って特徴量を作っていたのでそれに合わせて特徴量を登録します。(具体的な特徴量は書籍のリポジトリを参照してください)

```json
{
    "name": "pageviews",
    "template": {
        "function_score": {
            "field_value_factor": {
                "field": "pageviews",
                "missing": 0
            }
        }
    }
}
```

9200番ポートにホストしているElasticsearchへの特徴量追加リクエスト

```bash
curl -X POST -w '\n' -H 'Content-Type: application/json' -d @exmample_featureset.json "http://localhost:9200/_ltr/_featureset/example_featureset.json"
```

XGBoostを学習するのに用いるデータセットをElasticsearchから取得する際にはこの特徴量セットを指定してデータを取得、モデルの学習を行います。


### XGBoostの学習とモデルのアップロード

```json
GET /simplewiki/_search
{
  "query": {
    "match": {
      "text": "$keyword"
    }
  },
  "size": 10,
  "_source": ["title"],
  // ここまでは上で説明したsearch queryと同じ
  "rescore": {
    "query": {
      "rescore_query": {
        "sltr": {
          "params": {
            "keywords": "$keyword"  // 特徴量計算に使うパラメータ
          },
          "featureset": "example_featureset.json"  // アップロードしたfeatureset名
        }
      },
      "rescore_query_weight": 0.0
    },
    "window_size": 10
  },
  "ext": {  // 出力に特徴量を含める
    "ltr_log": {
      "log_specs": {
        "name": "log_entry0",
        "rescore_index": 0
      }
    }
  }
}
```

上の `$keyword` の部分をデータセットに含めるキーワードに変えてリクエストを行うことでデータセットを作成します。この処理についてはPythonのElasticserach APIを用いて行いました。データセットの正解ラベルですが、ここでは検索したユーザーが自身の検索語と完全に一致するタイトルを持つ記事をクリックするという強い仮定をおいて1,0でラベル付を行っています。

取得したデータセットを使ってモデルを学習したあとは、モデルをjson形式でダンプしてElasticsearchにアップロードします。

```bash
curl -X POST -w '\n' -H 'Content-Type: application/json' -d @xgb_model.json "http://localhost:9200/_ltr/_featureset/example_featureset.json/_createmodel"
```

:::details XGBoostの学習パラメータ

```python
params = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_leaves": 60,
    "subsample": 0.45,
    "eta": 0.1,
    "seed": 0,
}
```
:::

## 実験

### XGBoostの性能とシステムパフォーマンスへの影響

XGBoostを組み合わせたリランキングでは、リクエストを受け取ったあとに以下のような2段の処理が行われてレスポンスが返されます。

1. Elasticsearchのスコアを計算して上位N件を取得
2. N個の計算結果のそれぞれについてXGBoostを用いてスコアを計算し直す
3. リスコア結果をもとに並び替える

これはよく推薦システムで使われる候補集合の作成+リランクという2-stageシステムの構造と同じになっています。私の所属する企業のTech blogで2-stage推薦システムについての記事があるので興味のある方はどうぞ。

https://www.wantedly.com/companies/wantedly/post_articles/538673

BM25の性能よりXGBoostのほうが性能は良いとおそらく期待できますから、`N` は大きいほどステップ1での取りこぼしがなくなってランキングの性能は良くなると想像されますが、逆にXGBoostのスコアを計算する対象が増えるのでパフォーマンスには確実に悪影響を及ぼします。これがどの程度になるのかを調べるために以下のような指標を計算しました。

- nDCG@10
- latency
    - 平均値
    - 99%ile

結果は以下です。baselineは素のElasticsearchのスコアを使って並び替えをした結果です。

| model | N | nDCG@10 | latency (Avg.) | latency (99%ile) |
| --- | --- | --- | --- | --- |
| baseline | - | 0.7165 | 3.534 | 12.0 |
| xgboost | 100 | 0.7416 | 6.514 | 24.01 |
|  | 50 | 0.7411 | 6.702 | 23.02 |
|  | 20 | 0.7369 | 5.696 | 18.01 |
|  | 10 | 0.7381 | 4.988 | 17.01 |

XGBoostを用いたほうが性能は確実に向上していることがわかりますが、同時にパフォーマンスも落ちていて、N = 100としたときにはbaselineと比べてlatencyが倍ほどにまで悪化しています。

latencyの悪化についてもう少し詳しく見てみます。下の図は、検索に使ったキーワードでヒットする記事の数の累積ヒストグラムです。テストデータにふくまるキーワードの個数は500個ありますが、そのうち120キーワードは10件以下の検索結果しか返さないので、`N` を大きくしてもパフォーマンス悪化に寄与しません。N = 10 → N = 100としたとき、XGBoostに入力されるデータ数は9倍ほどになると予想されるので、単純に見るとlatencyは9倍ほど大きくなりそうです。しかし、実際にはlatencyの増加は2倍にも行っておらず抑えられています。これは単純に内部でバッチ推論をしているせいなのかElasticsearch側で他の最適化しているのかわかりませんが、効率的な推論が行われていることがわかります。


![](/images/es_search_system/Figure_1.png)

:::details Nを増やしたときにXGBoostが処理するデータ数の概算

あるキーワードkをElasticsearchに投げてヒットする文書の数を $h_k$ とします。Elasticsearchでとってくる候補の数を`N` としたとき、XGBoostに入力として与えられるデータ数 $d_w$ は

$$
    d_N = \sum_{k} \min(h_k, N)
$$

と表せます。ヒットする記事の件数がN件以上のキーワード数をn、N件未満しかヒットしないキーワードの集合を $k'$ とすると上の式は

$$
    \displaystyle d_N = Nn + \sum_{k'}h_{k'} \simeq Nn + \frac{500-n}{m}N
$$

ただし $m$ は適当な係数で、$N/m$ が $k'$ に含まれるキーワードのヒット数の重み付き平均となるようなものです。$m$を単純に2と仮定して概算すると $d_{100}/d_{10}\sim 9.3$となります。

:::

しかしここでの実験結果はElasticsearchのサーバーもリクエストもとのサーバーも同じLocalhostにある場合ですから、実運用の際にはlatencyの値自体は、同じデータセットを使ったとしても当然この値はもっと大きな値になります。この実験のようにユーザーからリクエストを受けて都度推論を回すようなオンライン推論を実現する場合はlatencyと性能のトレードオフをちゃんと考える必要がありそうです。

### Elasticsearch上での特徴量エンジニアリング

次に、Elasticserachのfeaturesetに新しい特徴量を追加するということを行ってみます。先述した `match_explore` は様々な統計量を計算することはできますが限界があります。自分で定義した特徴量を登録する方法がプラグイン側に用意されているのでそれを利用してみます。ここで追加する特徴量は以下の量です。

- 記事の本文の長さ
- クエリキーワードが記事本文に現れる回数

先程のfeaturesetを定義したjsonに、painlessという言語を使ってこれらの定義を追加します。

```json
{
    "name": "text_length",
    "template": {
        "function_score": {
            "script_score": {
                "script": {
                    "source": "params._source.text.length()",
                    "params": {
                        "_source": {
                            "includes": [
                                "text"
                            ]
                        }
                    }
                }
            }
        }
    }
},
{
    "name": "keyword_count_in_text",
    "params": [
        "keywords"
    ],
    "template": {
        "function_score": {
            "script_score": {
                "script": {
                    "source": "params._source.text.splitOnToken('{{keywords}}').length - 1",
                    "params": {
                        "_source": {
                            "includes": [
                                "text"
                            ]
                        }
                    }
                }
            }
        }
    }
}
```

作りたい特徴量はそこまで複雑なものではないはずですが、ネストもかなり深くなって書くのは結構大変でした(初めて書くせいもあるかと思いますが)。これを新しいfeaturesetとして登録して学習データセットを作成し、再度モデルの作成を行いました。その結果、nDCG@10は0.742と、特徴量前と比較して微増する結果になりました。

ElasticsearchをFeature store的に利用することは可能であることは自分でやってみてわかりましたが、追加コストが意外と高く、また(ちゃんと調べられていないだけかもしれないですが)あまり複雑な特徴量は登録できないのではないかと思いました。少し複雑な特徴量を作ろうとするとおそらくPython側でやったほうが簡単で、そうすると特徴量の置き場所が複数にバラけるといった状態になってしまうのではないかという懸念がありそうです。


## 感想

本を読みながら色々手を動かすことで、Elasticsearchを活用することで何ができて、逆に何がやりづらい・できないかを軽く把握することができました。色々考えながら試すことで例えば推論時にGPUを使うことは現時点ではできなかったり、推論に使えるモデルが限られていてLightGBMのモデルをjsonにダンプしてアップロードしてみても推論に使えなかったりと、活用の幅には限界がありそうだとわかったのは大きな収穫だったと思います。

一方で、本を読んで使ったElasticsearchの機能はほんの一部に過ぎないと思いますが、大量のデータから欲しい情報を柔軟に取り出すという意味でElasicsearchは便利に使うことができるということも理解できたと感じています。