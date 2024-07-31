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

詳しくない人向けに簡単に言うと、K9 AKMはDAC(Digital-Analog-Converter)で、デジタル信号をアナログ信号に変換してイヤホンやヘッドホンに出力するためのデバイスです。

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

OSをSDカードにインストールするのには Raspberry Pi Imager を使いました。Imagerでは事前にパスワードやSSHのセットアップ、WiFiのセットアップができるのでここで済ませておきます。以降の作業はすべてMacからのSSHで行っています。

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

ちなみに曲の停止は "p"(=pause)、ターミナルに戻るには"q"です(重要。これがわからなくてVimから抜け出せない人と同じ気持ちになりましたw)。

## DACで音楽を流せる状態の実現

イヤホンジャックからは再生できるようになりましたが、このままではUSBでDACに接続してそこから流すということはできません。外部デバイスのカード番号とデバイス番号をMPDに設定してやる必要があります。

```console
$ aplay -l  # オーディオデバイス情報の確認

card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
  Subdevices: 8/8
  Subdevice #0: subdevice #0
  Subdevice #1: subdevice #1
  Subdevice #2: subdevice #2
  Subdevice #3: subdevice #3
  Subdevice #4: subdevice #4
  Subdevice #5: subdevice #5
  Subdevice #6: subdevice #6
  Subdevice #7: subdevice #7
card 1: K9 [FiiO K9], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 3: vc4hdmi1 [vc4-hdmi-1], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

出力を確認すると、出力したい先のデバイス(FiiO K9)はカード番号１、デバイス番号０であることがわかりました。この情報を使って設定を追記します。

```bash
# /etc/asound.conf
pcm.!default {
    type hw
    card 1
    device 0
}

ctl.!default {
    type hw
    card 1
}
```

```bash
audio_output {
    type "alsa"
    name "My ALSA Device"
    device "hw:1,0"
    mixer_type  "hardware"
}
```

設定が終わったらデーモンを再起動します。

```console
sudo systemctl restart mpd
```

これで完了です。先程同様にncmpcppを起動してUSBで接続したDACから音が再生されることを確認します。

### 余談： 流せる曲のサンプリングレート

少し脇道にそれますが、音楽ファイルはファイル形式などによって異なる情報量(サンプリングレートやビット深度)を持つバージョンが複数あることが多いです。

- 圧縮音源 128kbps・320kbpsなど
- ロスレス音源(=CDと同等) 44.1kHz/16bit (=1,410kpbs)
- ハイレゾ音源
  - 48kHz/24bit
  - 96kHz/24bit
  - 192kHz/24bit
  - ...

有名どころだとSpotifyが2024/7現在で最高320kbps、Apple MusicやAmazon Musicではハイレゾまで対応していたりしますね。Androidだと14より前のバージョンではSRCという機能がついていて、96kHz/24bitより上の音源はすべて96kHz/24bitに落とされてしまうという仕様がありました(14以降は回避策が用意されています)。ぶっちゃけ96kHz/24bitより高品質な音源などは聴いても自分の耳では差などわからないのですが(小声)、ラズパイではそのような機能はないと思いつつ、気になったのでいろいろな音源を再生して確認してみました。

ビット深度の方は確認する手段を持ち合わせていなかったのですが、サンプリングレートの方は手持ちのUSB-DAC (FIIO KA17)に再生している音源のサンプリングレートを表示する機能があったのでそれを使います。KA17はPCM768kHz/32bitまで対応しているので今回の検証には十分です。検証に使う音源は[レコチョクのサイトで提供されているもの](https://recochoku.jp/hiresSample)からいくつかピックアップしました。試してみた結果、192kHzまでちゃんとラズパイから出力されており、少なくともこの領域まではちゃんと再生できることが確認できました。

## スマホから再生できる環境を手に入れる

ここまでで、ひとまず音楽を満足に再生することができる環境は揃いました。しかし、音楽を流そうと思ったら

- ラズパイにSSHする
- ncmpcppを起動する
- 流す

というステップがあり面倒です。また、ncmpcppは(まだ慣れていないせいもありますが)癖が強く、もっと操作性が良くできれば良いと感じていました。