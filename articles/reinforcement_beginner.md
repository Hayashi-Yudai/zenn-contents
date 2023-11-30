---
title: "強化学習における学習安定化の工夫を試してみた"
emoji: "📑"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["強化学習", "機械学習", "Python"]
published: true
---

機械学習スタートアップシリーズから出ている「[Pythonで学ぶ強化学習](https://www.amazon.co.jp/dp/4065142989)」という本を読んで強化学習に入門してみました。実際に自分で手を動かして学んだことなどを書いていきたいと思います。

## 実験環境

- MacBook Pro (M1 Max)
- Python 3.11

## 題材

この記事では、OpenAI Gymで提供されている倒立振子問題(CartPole)を題材として実験を行いました。このタスクに対してどのようにAgentを訓練することでより高い報酬、そして学習の安定性を達成できるのかという視点から実験をしてみました。

## 実験準備

### コードの構成

実装の登場人物は以下の3つです。

- Agent
- Environment
- Trainer

これら3つの関係性は下図のようになっていて、Agentが状況をもとにどう動くかを判断し、EnvironmentではAgentが取った行動をもとに次の状態や報酬を返します。そしてTrainerが上記2つのコンポーネントを取りまとめる役割を持っていて、Agentの学習を進めていくといった形になっています。

![](/images/reinforcement_beginner/reinforcement.png)

### Environmentの実装

Environmentの実装は非常にシンプルです。`gym.Wrapper` クラスを継承して、アクションを受け取ったときの返り値だけアレンジしています。`torch.Tensor` で返すようにしているのは、Agentにおける行動の価値関数の推定モデルにNNを使うことを想定しているためです。

```python
from typing import Any

import gym
import torch

class CartPoleObserver(gym.Wrapper):
    def reset(self):
        """環境をリセットして初期状態を返す"""
        return self.transform(super().reset())

    def step(self, action: int) -> tuple[Any]:
        """環境でactionを実行して、次の状態、報酬、終了フラグ、追加情報を返す"""
        n_state, reward, done, info = super().step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state) -> torch.Tensor:
        return torch.Tensor(state).reshape((1, -1))
```

### Agentの実装

Agentは内部に今の状態から次に起こす行動の価値を判断するモデル(`self.model` で定義されている)を持っていて、それを元につぎにどのような行動を取るべきか判断(`policy` メソッド)します。

```python
class ValueFunctionAgent:
    def __init__(
        self,
        epsilon: float,
        actions: list[int],
        gamma: float,
        batch_size: int,
        epsilon_decay: float = 1.0,
    ):
        self.epsilon = epsilon
        self.actions = actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay

        self.model = nn.Sequential(
            nn.Linear(4, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
        )
        self._teacher_model = deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def policy(self, s: torch.Tensor) -> int:
        """価値関数を使って行動を決定する"""
        if np.random.random() < self.epsilon:
            return torch.randint(len(self.actions), (1,)).item()

        estimates = self.estimate(s)
        return torch.argmax(estimates).item()

    def estimate(self, s: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            estimated = self.model(s)[0]
        return estimated

    def update(self, experiences: list[Experience]):
        experience_batch = random.sample(
            experiences, min(self.batch_size, len(experiences))
        )

        states = torch.vstack([e.state for e in experience_batch])
        n_states = torch.vstack([e.next_state for e in experience_batch])

        with torch.no_grad():
            estimateds = self.model(states)
            future = self._teacher_model(n_states)

        for i, e in enumerate(experience_batch):
            reward = e.reward
            if not e.done:
                reward += self.gamma * torch.max(future[i])
            estimateds[i][e.action] = reward

        self.model.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, estimateds)
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

    def update_teacher(self):
        self._teacher_model.load_state_dict(self.model.state_dict())
```

`update` メソッドは行動価値の推定モデルを学習するためのモデルです。後述するTrainerで管理しているAgentの経験を受け取って価値推定モデルのアップデートを行います。Agentでは実験のためにミニマルなものに加えていくつか工夫を加えています。

- Epsilon decay
- Fixed Target Q-Network

Epsilon decayはその名前の通り、Agentの取る行動のうち探索に使う割合を制御するパラメータであるepsilonを学習が進むに連れて小さくしていくものになっています。学習の初期ではAgentに様々な経験を蓄積させる必要があるので、モデルが最善と判断する行動だけではなくランダムな行動を取ることが有効ですが、学習が進むに連れてモデルが推定した最善の行動を取ったほうが学習効率的には高く、安定すると言われています。

Fixed Target Q-Networkは、Agentが取った行動による遷移先の状態の価値を推定するモデルを一定期間固定するテクニックです。これを使うことにより学習を安定させる効果があることが知られています。これはDeep Q-Networkの学習を安定させるために提案された手法であるらしいのですが、今回上のような浅いモデルでも効果があるのか気になって実験してみることにしました。

### Trainerの実装

```python
class ValueFunctionTrainer:
    def __init__(
        self,
        agent: ValueFunctionAgent,
        env: CartPoleObserver,
        buffer_size: int = 1024,
        report_interval: int = 10,
        teacher_update_interval: int = 3,
    ):
        self.agent = agent
        self.env = env

        self.buffer_size = buffer_size
        self.report_interval = report_interval
        self.teacher_update_interval = teacher_update_interval
        self.logger = Logger()
        self.training = False
        self.reward_log = []

    def train_loop(self) -> int:
        state = self.env.reset()
        done = False
        step_count = 0
        while not done:
            action = self.agent.policy(state)
            n_state, reward, done, _ = self.env.step(action)
            e = Experience(state, action, reward, n_state, done)
            self.experiences.append(e)
            if not self.training and len(self.experiences) == self.buffer_size:
                self.training = True

            if self.training:
                self.agent.update(self.experiences)

            state = n_state
            step_count += 1

        return step_count

    def train(self, episode_count=220):
        self.experiences = []
        self.training = False
        self.reward_log = []

        for i in range(episode_count):
            step_count = self.train_loop()
            if i % self.teacher_update_interval == 0:
                self.agent.update_teacher()
            self.episode_end(i, step_count)

    def episode_end(self, episode_num: int, episode_length: int):
        rewards = [e.reward for e in self.experiences[-episode_length:]]
        self.reward_log.append(sum(rewards))

        if episode_num > 0 and episode_num % self.report_interval == 0:
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode_num)
```

Trainerは先程説明したように、EnvironmentとAgentを取りまとめるような位置づけになっています。Environmentから現在の状態を受け取ってAgentに渡し、次に取るべき行動を決定、その行動をEnvironmentに渡して状態を遷移させます。Agentが取った行動による結果も `self.experiences` として保持していて、それをAgentに渡して行動価値推定モデルのアップデートを指せる役割も持っているので、いつモデルパラメータの更新をするか、という点もTrainerの責務です。

ここで `Experience` クラスは以下のような要素を持つdataclassになっています。

```python
@dataclass
class Experience:
    state: np.ndarray  # [position, acceleration, angle, angular_velocity]
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
```

## 実験結果

### NNを使っただけの実験(baseline)

まずは前述したEpsilon decayやFixed target Q-Networkを使わずに、単純にNNを使っただけの実験を行いました。

```python
# Fixed Target Q-Networkを除くのは頑張って上述の実装から関連部分を消す
def main():
    env = CartPoleObserver(gym.make("CartPole-v1"))
    agent = ValueFunctionAgent(
        epsilon=0.1,
        actions=list(range(env.action_space.n)),
        gamma=0.9,
        batch_size=64,
        epsilon_decay=1,  # decayさせない
    )
    trainer = ValueFunctionTrainer(agent=agent, env=env)

    trainer.train(episode_count=400)
    trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
```

400エピソード(ポールが倒れてやりなおしになるまでが1エピソード)実験してみて、10エピソードごとの報酬をプロットした結果が以下です。

![](/images/reinforcement_beginner/baseline2.png)

何回か実験してみたのですが、400エピソードでは報酬が飽和しなかったり、飽和したとしてもその飽和値が150~300くらいとぶれるような振る舞いを示していました。

### Epsilon decay

次にepsilon decayを入れたときの実験をしてみました。今回の実験では decayを0.98、つまりモデルのアップデートごとに探索に使う割合を2%ずつ落としていくようにしました。

実験結果は下のようになっていて、decayを入れることによって報酬の飽和値が350付近で安定するようになった気がします(何回か実験を回してみた感覚)。

![](/images/reinforcement_beginner/epsilon_decay_098.png)

### Fixed Target Q-Network

最後に、Fixed Target Q-Networkです。実験では、3回ごとに遷移先の状態での行動価値を推定するモデルを更新するようにしました。前述のepsilon decayと組み合わせるパターンと独立で使うパターンの2通りを試してみたのですが、独立で使っていたときにはとくに改善しているのかはわかりませんでした。一方で組み合わせて使ったときには報酬は500で最大値まで達している実験も出てきて、効率的な学習が進んでいるように見えました。

![](/images/reinforcement_beginner/fixed_q_098.png)

## 感想

強化学習を勉強してみたくて本を読んで、紹介されている手法をぱっと試してみました。強化学習でどのような学習の工夫が考えられているのかを色々学べ、簡単なモデルで試せたのは良かったと思っています。ただ手法の有効性について定量的に判断するというところまで出来なかったのは心残りなので時間があるときにまた試してみたいと思いました。