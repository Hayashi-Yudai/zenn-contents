---
title: "Elasticsearchを用いた検索ランキング作成"
emoji: "🤖"
type: "tech"
topics: ["Elasticsearch", "検索", "機械学習"]
published: false
---

私は業務で推薦システムの開発を行っているですが、最近検索周りで「[機械学習による検索ランキング改善ガイド](www.amazon.co.jp/dp/4814400306)」という評判が良い本が出版されたのでそれを読んで勉強しつつ、手を動かしてみたのでわかったことを紹介してみようと思います。

## この記事で話すこと

- Elasticsearchを使った検索ランキングの作成
- XGBoostを組み込んだリランキング
- Elasticsearch上での特徴量管理