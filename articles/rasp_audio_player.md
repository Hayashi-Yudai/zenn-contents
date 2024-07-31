---
title: "ラズパイでオーディオプレイヤー"
emoji: "🔥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["RaspberryPi", "Audio", "MPD"]
published: false
---

RaspberryPiを使ってオーディオプレイヤー的なものを作ってみたので、どんなものをどうやって作ったのかをせっかくなので書いてみようと思います。

## 環境

- Raspberry Pi 4
  - OSは Raspberry Pi OS Lite (64bit)
- SDカード (128 GB)
- USB AtoCケーブル
- FIIO K9 AKM

詳しくない人向けに簡単に言うと、K9 AKMはDAC(Digital-Analog-Converter)で、デジタル信号をアナログ信号に変換してイヤホンやヘッドホンに出力するための機械です。

## 簡単なSummary

- 何作ったの？
  - ラズパイのUSB端子からオーディオデータを出力して、DACを通して聴けるような仕組みを作った
  - ラズパイからの音楽再生・停止などの操作をスマホから行えるようにした
- どうやったの？
  - MPD (Music Player Daemon)をRaspberry Piに入れた
  - MPDの設定を変更して出力先としてDACを指定することでDACを通して音楽再生
  - MPDを同じネットワーク内に公開して、ラズパイ外部から触れるようにした。Androidスマホにクライアントアプリを入れてそこから操作可能にした

## 下準備

このあたりはサクッと行きます。

作業を始める前に下準備としてOSのセットアップをします。使い方としてはスマホやPCなどから操作して音楽を流せる状態を考えていたため、GUIは必要ないと判断しました。なのでOSとしては必要最小限のものが用意されている "Raspberry Pi OS Lite" を利用しました。これはイメージのサイズが0.4 GBととても軽量です。

OSをSDカードにインストールするのには Raspberry Pi Imager を使いました。Imagerでは事前にパスワードやSSHのセットアップ、WiFiのセットアップができるのでここで済ませておきます。

## 音楽を流せる状態の実現

必要最小限の構成で始めたので、まずは音を流せるようにすることから始めなければいけません。WindowsやMacではなんらかのMedia Playerが標準で入っているのでこのあたりは普段はあまり意識することがないかもしれません。

![](/images/rasp_audio_player/mac_media_player.png){scale=0.3}
*Macに標準搭載されているプレイヤー(?)*