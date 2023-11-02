---
title: "Elasticsearchを用いた検索ランキング作成"
emoji: "🤖"
type: "tech"
topics: ["Elasticsearch", "検索", "機械学習"]
published: false
---

私は業務で推薦システムの開発を行っているですが、最近検索周りで「[機械学習による検索ランキング改善ガイド](www.amazon.co.jp/dp/4814400306)」という評判が良い本が出版されたのでそれを読んで勉強しつつ、手を動かしてみたのでわかったことを紹介してみようと思います。

## この記事に書くこと

- Elasticsearchを使った検索ランキングの作成
- XGBoostを組み込んだリランキング
- Elasticsearch上での特徴量管理

## 書かないこと

- 実験環境の準備方法、データセットのESへのアップロードなど

## 実験を行う検索システムの構成

書籍で使われているコードをベースとして色々と自分で実験を行いました。

https://github.com/oreilly-japan/building-search-app-w-ml/tree/master

(構成図を書く)

## 実験を通して知りたかったこと

- Elasticsearchでどうやって検索システムを構築するのか
- 機械学習モデルを組み込むことでどれくらい性能が向上するのか
- 機械学習モデルを組み込むことでどれくらいシステムへの負荷が増大するのか
- Elasticsearchで特徴量を管理するのはどのような体験なのか

## 実験環境

- Dataset: simplewiki-202109-pages-with-pageviews-20211001
- Elasticsearch v7.13.4
- Python 3.11.6

## 実験

### Elasticsearchへのリクエスト

- keywordを指定して結果を引っ張ってくる方法

### featuresetへの特徴量のアップロード

- プラグインの説明
- 追加した特徴量
- アップロード方法

### XGBoostの学習とモデルのアップロード

- データセットへのラベルの付け方
- 作った特徴量を合わせてESから引っ張ってくる方法

### 評価

- 評価指標
    - nDCG@10
    - latency (Avg., 99%ile)
- 実験条件
    - window_size

### Elasticsearch上での特徴量エンジニアリング

- 追加した特徴量
- 実装方法
- 性能向上

## 感想

- GPU使えない
- XGBoostなど特定のモデルしか対応していない
- 結構特徴量追加するの大変
- ES上では作れない特徴量もできるはず、棲み分けをどうするのが正解なのはわからない