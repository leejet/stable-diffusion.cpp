# build.yml 差分メモ（本家 sd.cpp との差分）

このファイルは、fork 側の `build.yml` に入れている「配布ライブラリ拡大」の差分を説明するためのメモです。
目的は、今後本家アップデートを取り込むときに、エージェントや人間が差分意図を短時間で再構成できるようにすることです。

## 1. この fork の差分目的

本家の配布 ZIP は主に実行向けバイナリ（shared library 中心）です。
この fork では、利用者（特に Rust/MSVC 連携）で必要になるライブラリを配布物に同梱するため、以下を追加しています。

- Windows: `stable-diffusion.dll` に加えて `stable-diffusion.lib`（import library）を同梱

※以前は Linux/macOS 向けに `libstable-diffusion.a` の追加も試みられましたが、GGMLなどの依存関係が含まれない中途半端な静的ライブラリとなってしまうため、動的リンクの利便性を損なわないよう削除（本家仕様への差し戻し）が行われました。Linux/macOS ではデフォルトで生成される `.so` / `.dylib` のみで十分に多言語からのバインディングが可能です。

## 2. 変更対象ファイル

- `.github/workflows/build.yml`

このメモ作成時点では、上記 1 ファイルのみを変更しています。

## 3. ジョブ別の拡張内容

### 3.1 windows-latest-cmake

`Pack artifacts` に以下を追加。

- `stable-diffusion.lib` を `build` 配下から再帰探索
- `_deps` 配下を除外して候補を絞る
- 見つかった `.lib` を ZIP 対象ディレクトリ（`build/bin` または `build/bin/Release`）へコピー
- 見つからない場合は明示的に fail

意図:

- Windows の動的リンク利用時（Rust/MSVCなど）に必要な import library を配布 ZIP に含める

### 3.2 windows-latest-cmake-hip

`Pack artifacts` に以下を追加。

- `stable-diffusion.lib` を `build` 配下から再帰探索
- 見つかった `.lib` を `build/bin/stable-diffusion.lib` としてコピー
- 見つからない場合は fail

意図:

- ROCm 向け Windows 配布でも `.dll + .lib` を一貫して提供する

## 4. 期待する配布内容（差分の成果物）

この fork の想定成果物は以下です。

- Windows ZIP:
  - `stable-diffusion.dll`
  - `stable-diffusion.lib`
- Linux/macOS ZIP:
  - shared library（従来どおり）

## 5. 本家取り込み時のマージ指針

本家更新を取り込むときは、以下を確認してください。

1. Windowsジョブの `Pack artifacts` ステップに `stable-diffusion.lib` の同梱が残っているか
2. ZIP 名は変えずに中身だけ拡張する方針が維持されているか
3. `.lib` 未検出時に fail するガードが残っているか

## 6. 差分が不要になる条件

本家 `build.yml` が将来的に次を標準提供した場合、この差分は縮小または削除できます。

- Windows 配布 ZIP に `stable-diffusion.lib` を標準同梱
