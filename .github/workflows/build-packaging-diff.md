# build.yml 差分メモ（本家 sd.cpp との差分）

このファイルは、fork 側の `build.yml` に入れている「配布ライブラリ拡大」の差分を説明するためのメモです。
目的は、今後本家アップデートを取り込むときに、エージェントや人間が差分意図を短時間で再構成できるようにすることです。

## 1. この fork の差分目的

本家の配布 ZIP は主に実行向けバイナリ（shared library 中心）です。
この fork では、利用者（特に Rust/MSVC 連携）で必要になるライブラリを配布物に同梱するため、以下を追加しています。

- Windows: `stable-diffusion.dll` に加えて `stable-diffusion.lib`（import library）を同梱
- Linux/macOS: shared 出力に加えて `libstable-diffusion.a`（static archive）を同梱

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

- Windows の動的リンク利用時に必要な import library を配布 ZIP に含める

### 3.2 windows-latest-cmake-hip

`Pack artifacts` に以下を追加。

- `stable-diffusion.lib` を `build` 配下から再帰探索
- 見つかった `.lib` を `build/bin/stable-diffusion.lib` としてコピー
- 見つからない場合は fail

意図:

- ROCm 向け Windows 配布でも `.dll + .lib` を一貫して提供する

### 3.3 ubuntu-latest-cmake

`Build` に以下を追加。

- 既存 shared ビルド後、`build-static` ディレクトリで static も追加ビルド
  - `-DSD_BUILD_SHARED_LIBS=OFF`

`Pack artifacts` に以下を追加。

- `build-static` から `libstable-diffusion.a` を探索
- `build/bin/libstable-diffusion.a` にコピーして ZIP 同梱
- 見つからない場合は fail

### 3.4 ubuntu-latest-cmake-vulkan

`ubuntu-latest-cmake` と同じ方針。

- shared + static の 2 系統をビルド
- `libstable-diffusion.a` を ZIP へ同梱

### 3.5 macOS-latest-cmake

`Build` に以下を追加。

- 既存 shared ビルド後、`build-static` で static を追加ビルド
- `CMAKE_OSX_ARCHITECTURES` は shared 側と同じ設定を維持

`Pack artifacts` に以下を追加。

- `libstable-diffusion.a` を `build-static` から探索して同梱

### 3.6 ubuntu-latest-rocm

`Build` に以下を追加。

- shared ビルド後、`build-static` を別途作成して static 追加ビルド
- ROCm 関連コンパイラ/ターゲット設定を shared 側と同等に引き継ぐ

`Pack artifacts` に以下を追加。

- `libstable-diffusion.a` を同梱

## 4. 期待する配布内容（差分の成果物）

この fork の想定成果物は以下です。

- Windows ZIP:
  - `stable-diffusion.dll`
  - `stable-diffusion.lib`
- Linux/macOS ZIP:
  - shared library（従来どおり）
  - `libstable-diffusion.a`

## 5. 本家取り込み時のマージ指針

本家更新を取り込むときは、以下を確認してください。

1. 各ジョブの `Pack artifacts` ステップに「追加ライブラリ同梱」が残っているか
2. Linux/macOS 系で `build-static` の追加ビルドが落ちていないか
3. ZIP 名は変えずに中身だけ拡張する方針が維持されているか
4. `.lib/.a` 未検出時に fail するガードが残っているか

## 6. 差分が不要になる条件

本家 `build.yml` が将来的に次を標準提供した場合、この差分は縮小または削除できます。

- Windows 配布 ZIP に `stable-diffusion.lib` を標準同梱
- Linux/macOS 配布 ZIP に `libstable-diffusion.a` を標準同梱
- もしくは shared/static を明示的に分離した公式配布体系が整備される

## 7. 注意点

- この差分は CI 時間を増やします（shared + static の追加ビルド分）。
- 配布サイズも増えます（`.lib`/`.a` 同梱分）。
- ただし、利用側のリンク互換性（特にツールチェーン差）を優先しています。
