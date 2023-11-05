---
title: "LangChainを使ってgitの差分からドキュメントを更新する"
emoji: "📄"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [LangChain, ChatGPT]
published: false
---

「[ChatGPT/LangChainによるチャットシステム構築\[実践\]入門](https://www.amazon.co.jp/dp/4297138395)」という本を読んで学んだ知識を使って、自分で簡単なツールを作ってみたのでそれについて紹介しようと思います。作ってみたツールは、`git diff` の結果を入力として、この差分によって更新が必要になるドキュメントを検知して書き換えるというものです。

## 動作例

この記事で紹介するツールの実際の動作例を最初に示します。

```diff
❯ dupdate --repo ../dummy_project --model_name gpt-4 --k 2

2023-11-05 15:22:42.471 | INFO     | __main__:main:123 - Using mode: gpt-4
2023-11-05 15:22:43.440 | INFO     | __main__:main:125 - Created DB
2023-11-05 15:22:43.672 | INFO     | __main__:main:130 - Update the following document: simple_calc.md
2023-11-05 15:22:43.672 | INFO     | __main__:main:133 - Asking to ChatGPT...
--- Current Document
+++ Updated Document
@@ -2,8 +2,9 @@
 
 `calc_type`, `a`, `b` を標準入力として受取り、指定の計算を行った結果を標準出力に返す。
 
-- `calc_type` str: "add"、もしくは "sub" を指定可能。
+- `calc_type` str: "add"、"sub"、もしくは "mul" を指定可能。
     - "add": 加算を行う
     - "sub": 減算を行う
+    - "mul": 乗算を行う
 - `a` int
 - `b` int
Do you want to apply this update? [y/N]: y
2023-11-05 15:23:16.010 | INFO     | __main__:main:130 - Update the following document: complicated_calc.md
2023-11-05 15:23:16.011 | INFO     | __main__:main:133 - Asking to ChatGPT...
No changes found in complicated_calc.md.
```

動作検証用に標準入力から2つの整数を受け取って四則演算をした結果を返すプロジェクトを用意し、その上で動作させたものです。プロジェクトをgit管理し、コードに差分がある状態でドキュメント更新をさせたときの実行結果が上です。このプロジェクトには更新が必要なドキュメント1つと、コードの差分に全く関係がないドキュメントを1つ用意しているのですが、更新の必要な方だけ正しく更新ができているという結果が得られました。

:::details [参考]入力として与えたdiff

```diff
diff --git a/dummy_project/main.py b/dummy_project/main.py
index 13b538b..a018c1b 100644
--- a/dummy_project/main.py
+++ b/dummy_project/main.py
@@ -1,7 +1,7 @@
 import click
 
 @click.command()
-@click.option("--calc_type", type=click.Choice(["add", "sub"]))
+@click.option("--calc_type", type=click.Choice(["add", "sub", "mul"]))
 @click.option("--a", type=int)
 @click.option("--b", type=int)
 def simple_calc(calc_type: str, a: int, b: int):
@@ -10,6 +10,8 @@ def simple_calc(calc_type: str, a: int, b: int):
             print(a + b)
         case "sub":
             print(a - b)
+        case "mul":
+            print(a * b)
         case _:
             raise ValueError("Unknown calculation type")
```
:::

## ソースコード

https://github.com/Hayashi-Yudai/docs_updater

ここで紹介するコードは↑のリポジトリで公開しています。履歴を見るとわかるのですが、このツールは過去にちょっと作っていたことがあったのですがだいぶ雑に作っていて、今回書籍を読んで学んだ知識を使って書き直したところ9割ほど書き直すことになりました。。。

## 動作環境

- OS: Ubuntu22.04
- Python 3.11
- LangChain 0.0.329

## ツールの構成

ツールの構成自体は非常にシンプルで、図のようにVector Store、Embedding API、Chat APIを組み合わせて目的の動作を実現させています。

![](/images/langchain_doc_updator/structure.png)

処理のフローについては以下のようになります。

1. リポジトリに存在するドキュメントのembeddingをVector Storeに入れておく
2. gitの差分と類似度が高いドキュメントtop-Kを取り出す
3. K個のドキュメントのそれぞれについて、gitの差分とドキュメント本文をChatGPTに投げてドキュメントを更新させる

3で実際にドキュメント更新をさせるときには、確実にドキュメントの文章だけを出力で得るためにFunction Callingを利用します。

## 実装

### ユーザーとのインターフェースの決定

まず最初に、ユーザーがツールを利用するときに何を入力として与えるかを整理しました。今回の場合には以下の5つをユーザーに求める設計にしています。

- ドキュメント更新をしたいリポジトリのパス
- ドキュメント更新をさせるのに使うモデル
- リポジトリ内でのドキュメントファイルへのパス
- コードの差分との類似度のtopいくつまでを更新を試みる対象とするか
- デバッグ用の情報を提示するか

いくつかはデフォルト値を設定してユーザーが毎回入力しなくても良いようにします。これをコードに起こすと下のようになります。

```python
import click

@click.command()
@click.option("--repo", default=None, help="The path to the repository.")
@click.option(
    "--docs_path", default="docs", help="The path to documents in the repository."
)
@click.option("--model_name", default="gpt-3.5-turbo", help="The model name to use.")
@click.option("--k", default=1, help="The number of documents to retrieve.")
@click.option("--debug", default=False, help="Whether to print debug information.")
def main(repo: str, docs_path: str, model_name: str, k: int, debug: bool):
    # (具体的な処理)


if __name__ == "__main__":
    main()
```

### Vector Storeの用意

ドキュメントファイルは一般的にはそこまで膨大な量にならないことが期待できることから、Chromadbを使って実行のたびにVector Storeを構成します。

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma


def get_current_docs(docs_dir: str) -> list[Document]:
    docs = os.listdir(docs_dir)
    docs_contents = []

    for doc in docs:
        with open(f"{docs_dir}/{doc}", "r") as f:
            docs_contents.append(
                Document(page_content=f.read(), metadata={"title": doc})
            )

    return docs_contents

def create_vector_store(docs: list[Document]) -> Chroma:
    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embedding=embedding)

    return db
```

### Gitの差分と類似度の高いドキュメントの抽出

コードの差分を得る部分については GitPython というライブラリを利用します。

```python
import git

r = git.Repo(repo)
tree = r.head.commit.tree
git_diff = r.git.diff(tree)  # `git diff` の結果を得る

retriever = db.as_retriever(search_kwargs={"k": k})  # topいくつまで取るか指定
context_docs: list[Document] = retriever.get_relevant_documents(git_diff)
```

### ドキュメント更新

コードの差分と変更が必要である可能性のあるドキュメントが用意できたので、これらを用いてドキュメント更新を行います。ドキュメント更新に用いたプロンプトは以下のようなものです。

```text
以下に示すgit diffの結果をもとに、ドキュメントで更新が必要な部分がある場合には全て探し出し更新し、その結果を返してください。

===git diff
(コードの差分)

===古いドキュメント: (ドキュメントファイル名)
(ドキュメント本文)
```

これをそのままChatGPTに投げてもいいのですが、返ってきたメッセージからドキュメント本文に該当する部分を抜き出すのは大変なので、Function callingを使うことでjsonとして得たい値を受け取れるようにします。返してほしいjsonのスキーマを `{doc_filename: ..., doc_content: ...}` という形に指定してみます。

```python
import openai

def get_updated_doc_json(
    git_diff: str, doc: Document, model_name: str
) -> dict[str, str]:
    functions = [
        {
            "name": "extract_json_from_updated_doc",
            "description": "変更後のドキュメントをJSON形式で返します。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_filename": {
                        "type": "string",
                        "description": "変更を行ったドキュメントのファイル名",
                    },
                    "doc_content": {
                        "type": "string",
                        "description": "変更後のドキュメントの内容",
                    },
                },
            },
        }
    ]
    query = (
        "以下に示すgit diffの結果をもとに、ドキュメントで更新が必要な部分がある場合には全て探し出し更新し、その結果を返してください。"
        + "\n\n===git diff\n"
        + git_diff
        + f"\n\n===古いドキュメント: {doc.metadata['title']}\n"
        + doc.page_content
    )

    response = openai.ChatCompletion.create(
        model=model_name,
        temperature=0,
        messages=[{"role": "user", "content": query}],
        functions=functions,
        function_call={"name": "extract_json_from_updated_doc"},
    )
    message = response["choices"][0]["message"]

    return json.loads(message["function_call"]["arguments"])
```

これにより次にドキュメントを更新すると気に使いやすい形で返答を得ることができるようになります。ここまで得られたら後はドキュメントを実際に書き換えて終わりです。

## 今回できなかったがやってみたいこと

ここまで紹介してきたツールは、今の状態では更新が必要か検査するドキュメントの個数を人間が与える必要がある設計になっています(標準入力で `k` という名前で与えるもの)。ここは理想的にはツール内で更新の必要なドキュメントを必要な数だけ取ってきてほしいという思いがあります。これは `RetrievalQA` とChromadbを組み合わせれば行ける気がして試してみたのですが結果はうまく行きませんでした。

```python
retriever = db.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model_name=model_name, temperature=0.0)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
res = qa.run(
    f"以下のコードの差分によって変更が必要なドキュメント(.md, .rst)のファイル名を全て出力してください。\n\n==\n{git_diff}"
)

schema = {"properties": {"files": {"type": "array", "items": {"type": "string"}}}}
chain = create_extraction_chain(schema, llm)  # resからファイル名をリストとして抽出するタスク
print(chain.run(res))
```

これを実行しても、「変更の必要のあるドキュメントのファイル名はありませんでした」という結果が返ってきます。表現が気になってChatGPTに投げられている文章を確認してみたのですが、Chromadbから取得した以下のようなドキュメント情報のうち、ChatGPTに投げられているのは `page_content` だけになっているのが原因になっていそうなことがわかりました。

```python
Document(
    page_content="ドキュメント本文",
    metadata={"title": "ドキュメントのタイトル"}
)
```

:::details デバッグログ

```python
> /home/yudai/Documents/Python/docs_updater/.venv/lib/python3.11/site-packages/langchain/chains/llm.py(108)_call()
-> response = self.generate([inputs], run_manager=run_manager)
(Pdb) inputs
{
    'question': '以下のコードの差分によって変更が必要なドキュメント(.md, .rst)のファイル名を全て出力してください。\n\n==\ndiff --git a/dummy_project/main.py b/dummy_project/main.py\nindex 13b538b..a018c1b 100644\n--- a/dummy_project/main.py\n+++ b/dummy_project/main.py\n@@ -1,7 +1,7 @@\n import click\n \n @click.command()\n-@click.option("--calc_type", type=click.Choice(["add", "sub"]))\n+@click.option("--calc_type", type=click.Choice(["add", "sub", "mul"]))\n @click.option("--a", type=int)\n @click.option("--b", type=int)\n def simple_calc(calc_type: str, a: int, b: int):\n@@ -10,6 +10,8 @@ def simple_calc(calc_type: str, a: int, b: int):\n             print(a + b)\n         case "sub":\n             print(a - b)\n+        case "mul":\n+            print(a * b)\n         case _:\n             raise ValueError("Unknown calculation type")\n ', 
    'context': '## Simple calculation\n\n`calc_type`, `a`, `b` を標準入力として受取り、指定の計算を行った結果を標準出力に返す。\n\n- `calc_type` str: "add"、もしくは "sub" を指定可能。\n    - "add": 加算を行う\n    - "sub": 減算を行う\n- `a` int\n- `b` int\n\n\n## Complicated calculation\n\n`simple_calc`よりも高度な計算を行う。2つの複素数a, bを入力として受取り、その和を標準出力に返す。\n\n- `a_real` int: 変数aの実部\n- `a_imag` int: 変数aの虚部\n- `b_real` int: 変数aの実部\n- `b_imag` int: 変数aの虚部\n'
}
(Pdb)
```
:::

ここはパッと直せなかったのでまた今度実装してみようと思います。

## 感想

前回このツールを作ったときは、愚直にコードの差分とドキュメントファイル名を投げて更新が必要なファイルを名前だけから判断させて、、、ということをやっていたのですが、今回Vector Storeを使うことによってそのあたりをきれいに書くことができようになったのはかなり良い体験でした。また、後段のドキュメント更新の部分では、どうしてもChatGPTが余計なことをしゃべるせいでプロンプトで「ドキュメントファイルに書き込むこと以外はしゃべるな！」ということを試行錯誤して伝えようとして苦労していたのですがFunction callingによってそのような苦労をしなくて良くなったのは感動しました。

読んだ書籍の発行日がこの記事を書いている時点の3週間弱ほど前なのですが、書籍で使っているLangChainのバージョンと、この記事を書いている時点での最新バージョンが結構離れていて、LLM周りの進歩の速さを改めて実感しています。これからもキャッチアップを頑張っていかねば。。。