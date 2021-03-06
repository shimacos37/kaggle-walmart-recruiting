# kaggle-walmart

## 使用環境

- PC
```
MacBook Pro (16-inch, 2019)
プロセッサ 2.3 GHz 8コアIntel Core i9
メモリ 32 GB 2667 MHz DDR4
```
- 環境構築
    - gcloud SDK
        - Installation guide for Mac OS https://cloud.google.com/sdk/docs/quickstart-macos
        - Authentication:
            - `gcloud auth application-default login`
    - pyenv
        ```sh
        git clone git://github.com/yyuu/pyenv.git ~/.pyenv
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc [or your configuration file (ex. ~/.zshrc)]
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        source ~/.bashrc
        ```
    - poetry
        ```sh
        curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
        source $HOME/.poetry/env
        ```
- ライブラリのinstall
    ```sh
    pyenv install 3.7.7
    pyenv local 3.7.7
    poetry config virtualenvs.in-project true
    poetry install
    ```
## 諸々の考察

- EDAとそれに対する考察、ベースラインについては`./notebooks/baseline.ipynb`を参照

- 8ヶ月分を予測する必要がある
  - リアルタイムデータを使えないので、一気に予測しても精度でなさそう。
    - 特に、8ヶ月後なんかは統計値の方が当たる可能性がある。
  - 一旦統計値でベースラインを作ってみる (`./notebooks/baseline.ipynb`を参照)。
    - Public: 3162.79282 (Private: 3217.38001)
    - このスコアをベースにして考えていく。 
- 使用手法
  - 今回のような長期間の予測をしたい場合は、ARIMAなどの自己回帰モデルやSeq2Seqなどの手法を用いることが考えられる。
    - ARIMA: 特徴量などの作成がしにくく精度は出なそうだが、統計値よりはいいか
    - Seq2Seq: 実装コストが高い。kaggleなら試すが、今回は簡単な課題なので一旦保留。
    - 今回は、精度の面で実績があり、実装コストの低いLightGBMを採用する。
      - ただし、8ヶ月後などをちゃんと予測できるようにするにはある程度工夫が必要になる。
      - また、lightGBMは過学習しやすいため、木の深さは浅めにパラメータを設定する
- 方針
  - target特徴が効くというのは経験的にあるので、それを使うことを前提にどうfoldを切るかを考える。
  - 大まかに二つ方法があると考えられる。
      1. 月単位のエキスパートモデルを作成
      2. 何ヶ月後を予測するかのエキスパートモデルをいくつか作成
  - 1の場合の方が月の特徴を捕らえられそう？
    - 11月や12月の予測が重要だと考えられるため、重要そう。
  - 2の場合の方が直近のトレンドが重要な場合が良さそう。
    - 2でも1年前の月特徴何かを加えることで1でしたいことは実現できそう + 重要そうな11,12がtest期間の序盤にあるため両方考慮できそうな2でとりあえず進める
    - 特徴量作成で気をつけなければいけないのは、例えば1ヶ月後を予測したいとする時に、各時点から直近1ヶ月のデータを見えなくさせること。
      - このようなwindow関数系の特徴量はpandasだと作成しにくい (特に`RANGE BETWEEN`系)ので、BigQueryで特徴量作成を行う。
  - validationは計算時間短縮の観点で3foldで、一般的なTimeSplitCVで訓練終端から1ヶ月、2ヶ月、3ヶ月の3つで行う。

  

  ## 結果

| model                    	    | Public     	| Private    	| CV (1ヶ月後予測モデル)| CV (2ヶ月後予測モデル)
|--------------------------	    |------------	|------------	|------------	     |------------	     |
| 統計値(4週目を3週目にNormalize)  | 3162.79282 	 | 3217.38001  | 
| 統計値(12月に関してNormalize削除)| 3498.14377 	 | 3625.00550  | 
| lightgbm (1ヶ月毎で予測) (v1)   | 3275.45756 	 　 | 3408.86372  | 1686.51 | 1607.69
| + lag特徴量 (v2)               | 3130.00770      | 3266.38715  | 1323.52 | 1403.17
| + 統計値と単純平均               | 2784.22291      | 2873.88102  | 
| v2 + 差分特徴 (v3)              | 3165.57918      | 3290.99826  | 1312.28 | 1356.15
| v2 + postprocess              | 2997.41326      | 3115.74900  | | 
| v2 + postprocess +　統計値と平均 | 2782.64669      | 2861.27539  | | 
| v3 + norm特徴 (v4)             | 3108.17901      | 3236.75181  | 1295.13 | 1353.58
| (v2 + v4 + 統計値 ) / 3         | 2771.07706      | 2856.65531  |  | 

### 結果からの考察

- 統計値で12月についてNormalizeを取り除くと改悪するのは、与えられているデータが週毎の集計値になっているため、年末の売り上げが下がる効果とクリスマスの売り上げが上がる効果が平均化されてしまっているからか？
  - 例えば、訓練データでは2010-12-24, 2010-12-31, 2011-12-23, 2011-12-30が存在しており、綺麗に年末とクリスマスが分かれている。一方、テストデータでは2012-12-21, 2012-12-28で集計されていて、訓練データに比べて2012-12-28の年末効果が薄く、2012-12-21のクリスマス効果がでかいのではないかと考えられる。
  - ここを調整するpostprocessを行うと良さそう。
  - 2012-12-21週を6/7倍、2012-12-28週を8/7倍にするpostprocessで精度向上を確認 (実務では使えないが。。)
    - お気持ちとしては、2012-12-21週は訓練期間よりも年末を含む期間が少ないので、訓練時期よりも売り上げが少なくなるはず。
    - 2012-12-28週は訓練期間よりも年末年始を含む期間が少ないので、訓練時期よりも売り上げが多くなるはず。
- 今回MAEが評価指標なので、mae最適化やmedianの統計値も試したがよくなかった。


## TIPS

### SQLについて

- SQLをjinja2テンプレートで作成することで、集約系の特徴量を簡単に書けるようにしてる
- 何ヶ月後を予測するためのパラメータ(`base_secs`)も外出しすることで、一つのSQLで全てのモデルに必要な特徴量を作成することができる。

### バージョン切り替え

- hydraを使うことで、`feature=v1`, `feature=v2`などと指定するだけで切り替えれるようにしている。

## 実行

- gcpのproject_idを`yamls/config.yaml`内のproject_idに書く
- BQのwalmartデータセット以下にcsvのファイル名と同じ名前のテーブルをアップロードして作成する。
- 特徴量の作成
```sh
# versionは上の表のものに準じている
poetry run python create_feature.py feature.version=1
```
- 訓練
```sh
# versionは上の表のものに準じている
sh bin/train_v{version}.sh
```
- 予測結果の統合・可視化は`notebooks/result_analysis.ipynb`で行った。

## その他アイデア

- validationの作り方が今だと直近の1-3ヶ月になってしまっているが、もう少しtest期間の月に合わせたsplitにしたい。
- パラメータ最適化
- ARIMAなどの統計modelやOverfitしにくいモデルとアンサンブル
- Markdown系の深堀り
- [Darts](https://github.com/unit8co/darts)でモデルを作成する
- 週毎の集計値を日毎の集計に直して学習する。
  - IsHoliday部分に重みをつけて7日に配分する。
## 所感

- 個人的には、8ヶ月分予測と聞いてまず統計値でベースラインを作れるかは大事な気がする。
  - 結果的に統計値に比べて優位に強いmodelをlightGBMで作れたわけではないが、ちょっと混ぜるだけで60位くらいには入れる。
- これをやった後にKernelを見たら、特徴量何も作らずにRandomForestに突っ込んで割と精度が出ててビビった。
  - 定常性がある分、何週目かとStore, Deptの情報だけで割と当てられるのかも知れない。
  - simpleな特徴だけで最初にmodelを作らなかったのは反省。
  - ただ現実に8ヶ月分予測する必要があるかと言われると結構微妙な気がする。。
