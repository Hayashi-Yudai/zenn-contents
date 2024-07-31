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
  - MPD (Music Player Daemon)とncmpcppをRaspberry Piに入れた
  - MPDの設定を変更して出力先としてDACを指定することでDACを通して音楽再生
  - MPDを同じネットワーク内に公開して、ラズパイ外部から触れるようにした。Androidスマホにクライアントアプリを入れてそこから操作可能にした

## 下準備

このあたりはサクッと行きます。

作業を始める前に下準備としてOSのセットアップをします。使い方としてはスマホやPCなどから操作して音楽を流せる状態を考えていたため、GUIは必要ないと判断しました。なのでOSとしては必要最小限のものが用意されている "Raspberry Pi OS Lite" を利用しました。これはイメージのサイズが0.4 GBととても軽量です。

OSをSDカードにインストールするのには Raspberry Pi Imager を使いました。Imagerでは事前にパスワードやSSHのセットアップ、WiFiのセットアップができるのでここで済ませておきます。

## 音楽を流せる状態の実現

必要最小限の構成で始めたので、まずは音を流せるようにすることから始めなければいけません。WindowsやMacではなんらかのMedia Playerが標準で入っているのでこのあたりは普段はあまり意識することがないかもしれません。

![](/images/rasp_audio_player/mac_media_player.png =400x)
*Macに標準搭載されているプレイヤー(?)で音楽が再生されている様子*

Linux上で再生させるにはどうすればいいのだろうと色々調べていて、どうやらRhythmboxというものが有名らしいし、rhythmbox-clientというものも用意されていてCLIから操作できそう！ということがわかってきたのですが、クライアントを動かすにはRhythmboxが動いている必要があり、Rhythmboxを動かすにはGUIが必要、ということがわかり断念しました。さらに調べていくと、MPD (Music Player Daemon) + ncmpcpp という組み合わせでやりたいことがシンプルに実現できそう、ということがわかってきました。MPDはその名の通り音楽を流すためのソフトウェアで、ncmpcppはMPDクライアントです。さっそく入れてみます。

```console
sudo apt-get update
sudo apt-get install mpd ncmpcpp
```

次に、初期設定です。MPDの設定は `/etc/mpd.conf`で行います。

```bash
# 自分の環境に合わせて変更
music_directory "/home/yudai/Music"

# MP3やFLACを再生するための設定
decoder {
    plugin "ffmpeg"
    enabled "yes"
}
```

ここまでできたらデーモンを起動します。

```console
sudo systemctl start mpd
sudo systemctl enable mpd
```

これで音楽を再生できる環境は整ったはずです。ncmpcppを起動して試しに流してみます。出力先はラズパイに付属しているイヤホンジャックです。

```console
ncmpcpp
```

"u"キーを押すとリストがアップデートされて、 `music_directory` 内のファイルが表示されるはずです。上下キーでカーソルを曲名に合わせてEnterで再生されます。

![](/images/rasp_audio_player/ncmpcpp_sample.png =500x)
*ncmpcppの実行画面*

ちなみに曲の停止は "p"(=pause)、ターミナルに戻るには"q"です。