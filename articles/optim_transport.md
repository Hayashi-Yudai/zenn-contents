---
title: "点群の最適輸送アルゴリズムで遊んでみた"
emoji: "🔥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "最適輸送問題", "線形計画法"]
published: true
---

この記事は、「[最適輸送の理論とアルゴリズム](https://www.amazon.co.jp/%E6%9C%80%E9%81%A9%E8%BC%B8%E9%80%81%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0-%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%83%95%E3%82%A7%E3%83%83%E3%82%B7%E3%83%A7%E3%83%8A%E3%83%AB%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E4%BD%90%E8%97%A4-%E7%AB%9C%E9%A6%AC/dp/4065305144/ref=tmm_pap_swatch_0?_encoding=UTF8&qid=1690814042&sr=8-1)」を読んでいて気になった部分を自分で実装して試してみたものを雑にまとめたものです。今回は点群の最適輸送について、実際にPythonで実装をして実験してみた結果を書きます。最適輸送理論は機械学習においてロス関数の設計に役立つらしい(本をまだ全部読んでない...)ので、シンプルな線形計画問題で解ける簡単なアルゴリズムを一度実装してみました。

## 最適輸送問題とは

まず、そもそも最適輸送問題って何？という話なのですが、

- 2つの分布$D_1$, $D_2$がある
- 分布を移動させるときのコストが定義されている

という状況で$D_1$を$D_2$に総コストを最小にして移動させるためにはどのようにしたらよいか、という問題だと理解しています。一般の連続分布ではなく点群の話に言い換えると、点群に含まれる点を移動コストを最小にして別の点に移動させるためにはどの点をどこの点に移動させればいいか、という問題です。

## 点群の最適輸送問題の定式化

点群Aを別の点群Bに移動させる問題を考えます。$C_{i,j}$を点群Aの$i$番目の点を点群Bの$j$番目の点に移動させる場合にかかるコストとします。$P_{i,j}$を点群Aの$i$番目の点を点群Bの$j$番目の点にどれくらい移動させるかを表す行列(輸送行列)とします。このとき、最適輸送問題は、

$$
\min_{P}\sum_{i}\sum_{j}C_{i,j}P_{i,j}
$$

と書くことができます。また制約条件としては、

$$
    P_{i,j} \ge 0
$$

$$
    \sum_{j}P_{i,j} = a_i
$$

$$
    \sum_{i}P_{i,j} = b_j
$$

の3つがあります。2番目と3番目の制約条件は質量保存の法則的なもので、輸送の過程で点群の質量が失われないという条件です。

## Pythonによる実装

上の定式化を見るとわかるように、これは制約付きの線形計画問題なのでScipyなどに用意されているソルバーを使って解くことができます。

```python
if __name__ == "__main__":
    start = np.array([[2.2, 2.1], [3.2, 5.3], [4.5, 4.4], [3.1, 3.8]])
    end = np.array([[4.8, 1.9], [4.1, 3.3], [2.0, 5.5], [3.4, 2.5]])

    a = np.ones(4) / 4
    b = np.ones(4) / 4

    C = calc_cost(start, end)

    P = solve_transport_problem(C, a, b)
    print("P: ", P)
    print("Minimum Cost: ", np.sum(P * C))

```

ここでは本と同じ例を使って解いてみます。`start` と `end` がそれぞれ輸送前と輸送のターゲット先の点群の座標です。分布の点の重みは等しく、総和が1になるようにしています(`a`, `b`)

```python
def calc_cost(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    cost = np.zeros((len(start), len(end)))

    for i in range(len(start)):
        for j in range(len(end)):
            cost[i, j] = np.linalg.norm(start[i] - end[j]) ** 2

    return cost
```

輸送コストの設計ですが、簡単のため点間の二乗距離で定義します。


```python
def solve_transport_problem(
    C: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    num_x = len(a)
    num_y = len(b)

    c = C.flatten()
    A = []

    # \sum_{j}P_{i, j} = a_i
    for i in range(num_x):
        A_i = np.zeros((num_x, num_y))
        A_i[i, :] = 1
        A.append(A_i.flatten())

    # \sum_{i}P_{i, j} = b_j
    for j in range(num_y):
        A_j = np.zeros((num_x, num_y))
        A_j[:, j] = 1
        A.append(A_j.flatten())

    A = np.array(A)
    b = np.concatenate([a, b])

    res = linprog(c, A_eq=A, b_eq=b, method="highs")
    P = res.x.reshape((num_x, num_y))

    return P

```

scipyの `linprog` メソッドを利用して解くと

```bash
P:  [[-0.    0.    0.    0.25]
 [ 0.    0.    0.25  0.  ]
 [ 0.25  0.    0.    0.  ]
 [-0.    0.25  0.    0.  ]]
Minimum Cost:  2.6675000000000004
```

が得られます。これは `start` の最初の点 `[2.2, 2,1]` は `end` の最後の要素の点 `[3.4, 2.5]` に移動させるのが最適ということを示してます。視覚的に表すならば、

![](https://storage.googleapis.com/zenn-user-upload/0a10e83e40a8-20230730.png)

のようになります(赤: `start`、青：`end`)。

:::details code

```python
def plot_transport(start: np.ndarray, end: np.ndarray, P: np.ndarray):
    plt.figure()
    plt.scatter(start[:, 0], start[:, 1], c="r", label="start")
    plt.scatter(end[:, 0], end[:, 1], c="b", label="end")

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                plt.annotate(
                    "",
                    xy=end[j],
                    xytext=start[i],
                    arrowprops=dict(arrowstyle="->", color="k", lw=1),
                )

    plt.show()

```
:::

### コスト関数を変えたときの挙動の変化

コスト関数の定義の仕方を色々変えてみたときに最適な点群の輸送がどう変わるか見てみます。

#### 逆二乗

![](/images/optim_transport/inverse_square.png)

ユークリッド距離が大きくなるほどコストが小さくなるような構造なので、なるべく離れた点に輸送が行われるようになっていることがわかります。


#### 非対称なコスト関数

$x$方向と$y$方向で非対称なコスト関数 $10\Delta x^2 + \Delta y^2$ を設計すると

![](/images/optim_transport/anti_symmetric_cos.png)

$x$方向に輸送するとコストが大きくなるので、$y$方向への輸送が多くなります。機械学習の文脈で考えて赤の点をモデルの推定値、青の点を正解の推定値として見てみると、コスト関数の設計はロス関数の設計に見えてきます。$x$方向と$y$方向に何らかの意味がある時、上のコスト関数の設計はある一方への間違え方($y$方向)はある程度許容するが、もう一方($x$方向)への間違え方は許容しないという思想を設計していることになります。

## 感想

この記事の内容は本の最初の10 %ぐらいの内容なのですが、もう既に残り90 %を理解しきれるか心配です。ただ、実際に数値例を使って計算してみることで機械学習にどのように使えるのか、なんとなくですが理解できた気がするのかな、と思っています。
