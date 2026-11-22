<h1 align="center">
  <img src="res/img/carquet_banner.png" alt="Carquet" width="840" />
</h1>

<p align="center">
  Une bibliothèque C pure et rapide pour lire et écrire des fichiers Apache Parquet.
</p>

<p align="center">
  <a href="https://vitruves.github.io/carquet/"><img src="https://img.shields.io/badge/Website-vitruves.github.io%2Fcarquet-2a78d6?style=for-the-badge" alt="Site web" /></a>
</p>

<p align="center">
  <a href="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml"><img src="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg" alt="Compilation" /></a>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue" alt="Plateforme" />
  <img src="https://img.shields.io/badge/C-C11-blue" alt="Norme C" />
  <a href="https://github.com/Vitruves/carquet/releases"><img src="https://img.shields.io/github/v/release/Vitruves/carquet?color=blue" alt="Version" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="Licence" /></a>
  <a href="https://github.com/Vitruves/carquet/graphs/traffic"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Vitruves/carquet/traffic/clones.json" alt="Clones" /></a>
  <br/>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/v/carquet?color=blue&label=pypi" alt="PyPI" /></a>
  <a href="https://pypi.org/project/carquet/"><img src="https://img.shields.io/pypi/pyversions/carquet" alt="Versions de Python" /></a>
  <a href="https://pepy.tech/projects/carquet"><img src="https://img.shields.io/pepy/dt/carquet?color=blue&label=pypi%20downloads" alt="Téléchargements PyPI" /></a>
  <img src="https://img.shields.io/badge/SIMD-SSE4.2%20%C2%B7%20AVX2%20%C2%B7%20AVX--512%20%C2%B7%20NEON%20%C2%B7%20SVE-eb6834" alt="SIMD" />
</p>

<p align="center">
  <a href="#points-forts">Points forts</a> &nbsp;·&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#configurations-de-compilation">Configurations de compilation</a> &nbsp;·&nbsp;
  <a href="#api-c">API C</a> &nbsp;·&nbsp;
  <a href="#api-python">API Python</a> &nbsp;·&nbsp;
  <a href="#outil-en-ligne-de-commande">CLI</a> &nbsp;·&nbsp;
  <a href="#fonctionnalités-parquet-prises-en-charge">Fonctionnalités</a> &nbsp;·&nbsp;
  <a href="docs/README.md">Manuel</a> &nbsp;·&nbsp;
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <a href="README.md">English</a> &nbsp;·&nbsp;
  <a href="README.zh-CN.md">简体中文</a> &nbsp;·&nbsp;
  <a href="README.ja.md">日本語</a> &nbsp;·&nbsp;
  <a href="README.ko.md">한국어</a> &nbsp;·&nbsp;
  <a href="README.de.md">Deutsch</a> &nbsp;·&nbsp;
  <b>Français</b> &nbsp;·&nbsp;
  <a href="README.pt-BR.md">Português (Brasil)</a>
</p>

<br/>

---

<!-- ────────────────────────────  POINTS FORTS  ──────────────────────────── -->

## Points forts

<div align="center">

| | |
|---|---|
| ⚡ **Rapide** | Écrit **2.3–10.6x plus vite** qu’Arrow C++ et lit les données compressées **1.5–2.1x plus vite** à partir du même fichier (Apple M3, 10 M de lignes) ; avec un codec, les lectures sont aussi **3.0–4.4x plus rapides** que celles de polars et **2.9–4.3x plus rapides** que celles de DuckDB. Les lectures non compressées sont servies directement depuis le mmap en **0.26 ms**. SIMD pour SSE4.2 / AVX2 / AVX-512 / NEON / SVE avec sélection à l’exécution. |
| 🪶 **Compacte** | C11 pur, une bibliothèque d’**environ 500 Ko** (Arrow C++ : plus de 50 Mo). Quatre dépendances de compression (zstd, zlib, lz4, brotli), récupérées automatiquement si elles manquent. Se compile en moins d’une minute avec un simple `make` ou `xmake`. |
| 📦 **Complète** | Tous les types physiques et encodages, Snappy / GZIP / LZ4 / ZSTD / Brotli, pages V1 + V2, schémas imbriqués à toute profondeur, filtres de Bloom, index de pages, élagage par statistiques, ajout. [Matrice complète ↓](#fonctionnalités-parquet-prises-en-charge) |
| 🔗 **Interopérable** | Fichiers lus par PyArrow, DuckDB et Spark sans configuration. Passerelle [Arrow C Data Interface](docs/reading.md#export-to-arrow-c-data-interface) sans dépendance (struct / list / map à toute profondeur) et une [API Python](#api-python) complète, sans dépendance. |
| 🛠 **Outillage** | CLI `carquet` pour inspecter, filtrer, exporter et valider des fichiers, plus `codegen` pour produire un lecteur C prêt à compiler. Un [skill pour agents](#ai-coding-agents) prêt à l’emploi pour Claude Code et Codex. |

</div>

<br/>

---

<!-- ────────────────────────────  BENCHMARKS  ──────────────────────────── -->

## Benchmarks

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="res/img/bench_10m_m3_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="res/img/bench_10m_m3_light.png">
    <img src="res/img/bench_10m_m3_light.png" alt="Temps d’écriture et de lecture pour 10 millions de lignes sur Apple M3 : carquet vs Arrow C++, PyArrow, fastparquet, polars et DuckDB, par codec" width="900">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="res/img/bench_10m_linux_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="res/img/bench_10m_linux_light.png">
    <img src="res/img/bench_10m_linux_light.png" alt="Temps d’écriture et de lecture pour 10 millions de lignes sous Linux (Intel Xeon D-1531) : carquet vs Arrow C++, PyArrow, fastparquet, polars et DuckDB, par codec" width="900">
  </picture>
</p>

<p align="center"><b><a href="https://vitruves.github.io/carquet/">Explorer le benchmark interactif sur le site web →</a></b><br/><sub>basculez entre lecture et écriture, et entre 100K, 1M et 10M de lignes</sub></p>

Accélération de carquet par rapport à Arrow C++ à 10 M de lignes (plus élevé = carquet plus rapide) :

<div align="center">

| | ARM · Apple M3 | | x86 · Windows (Haswell, MSVC) | | x86 · Linux (Xeon D-1531) | |
|---|---:|---:|---:|---:|---:|---:|
| **Codec** | **Écriture** | **Lecture** | **Écriture** | **Lecture** | **Écriture** | **Lecture** |
| snappy | **7.24x** | **2.68x** | **3.39x** | **1.17x** | **5.82x** | **2.65x** |
| zstd | **10.6x** | **3.14x** | **3.83x** | **1.13x** | **9.93x** | **2.18x** |
| lz4 | **6.01x** | **1.88x** | **2.96x** | **1.05x** | **5.66x** | **1.30x** |
| none | **2.32x** | **55.9x**\* | **2.10x** | **62.0x**\* | **3.55x** | **36.3x**\* |

</div>

<sub>\* Chemin mmap zéro-copie. Apple M3 : carquet 0.9.0 vs Arrow C++ 26.0.1 (2026-11-22). Linux : le même carquet vs Arrow C++ 23.0.1 (2026-11-22, fichiers sur tmpfs). Windows : carquet 0.7.3 vs Arrow C++ 24.0.0 (2026-11-02), non remesuré depuis. Avec un codec, carquet encode de lui-même par dictionnaire les deux colonnes d’entiers, ce qu’Arrow C++ a pour consigne de ne pas faire ici (voir la méthodologie).</sub>

Les lectures compressées décodent chaque valeur, et sous Linux et macOS les deux bibliothèques utilisent les mêmes lz4 / zstd système : la vitesse des codecs est donc identique. La comparaison la plus équitable est la **lecture croisée du même fichier** (les deux bibliothèques lisent exactement le même fichier) : carquet est **1.5–2.1x plus rapide** qu’Arrow C++ sur M3 (2.0–3.8x plus rapide que polars), 1.7–2.7x sous Linux (1.7–3.9x plus rapide que polars) et 1.04–1.30x sur la VM Windows, dont les lectures à froid sont limitées par le disque virtuel.

<details>
<summary><b>Méthodologie des benchmarks</b></summary>

<br/>

Tous les benchmarks utilisent des données identiques (PRNG LCG déterministe), les mêmes row groups, le même niveau zstd et la même taille de page, la médiane tronquée de 11 à 51 itérations, avec purge du cache de pages de l’OS entre les phases d’écriture et de lecture et un temps de refroidissement entre les configurations. Schéma : 3 colonnes (INT64, DOUBLE, INT32). Comparé au lecteur Parquet bas niveau d’Arrow C++ (sans matérialisation en Arrow Table), à PyArrow, à fastparquet, au lecteur et à l’écrivain natifs (Rust) de polars et à ceux, natifs (C++), de DuckDB.

Encodages : carquet écrit avec ses réglages par défaut, à savoir PLAIN sans codec et, avec un codec, BYTE_STREAM_SPLIT pour la colonne DOUBLE et un dictionnaire sur une colonne d’entiers lorsqu’un échantillon du chunk indique que c’est rentable (les deux colonnes d’entiers ici) ; les sommes de contrôle de pages sont activées. Arrow C++, PyArrow et fastparquet tournent sans dictionnaire (BYTE_STREAM_SPLIT pour les doubles là où ils savent l’écrire), comme dans les versions précédentes de ce benchmark, leurs fichiers compressés sont donc plus gros que ceux de carquet ; polars et DuckDB choisissent eux-mêmes leurs encodages et encodent par dictionnaire les deux colonnes d’entiers, avec ou sans codec, ce qui explique que leurs fichiers non compressés soient 45% plus petits que ceux, en PLAIN, de carquet. Les tailles des fichiers figurent dans les tableaux.

Le cache de pages est purgé fichier par fichier : `posix_fadvise(DONTNEED)` sous Linux ; sous macOS et Windows, le fichier est copié sans mise en cache (`F_NOCACHE`, `FILE_FLAG_NO_BUFFERING`) et la copie le remplace (`benchmark/bench_purge_win32.h`). Sous Windows, carquet lie les zstd / lz4 qu’il compile lui-même et Arrow C++ ceux de conda-forge, la vitesse des codecs peut donc légèrement différer entre les deux.

Le benchmark de **lecture croisée du même fichier** est la comparaison la plus équitable : les deux bibliothèques lisent exactement le même fichier Parquet (écrit par l’une, lu par les deux). Cela élimine les différences de taille de pages, de choix d’encodage et de disposition des row groups.

Les **lectures non compressées** marquées d’un \* utilisent le **chemin mmap zéro-copie** de Carquet : pour les colonnes requises, de taille fixe, non compressées et encodées en PLAIN, le lecteur par lots renvoie des pointeurs directement dans le fichier mappé en mémoire, sans memcpy. Arrow matérialise toujours dans ses propres tampons. **Les chiffres de lecture compressée sont la mesure la plus représentative du débit de lecture de bout en bout.**

</details>

<details>
<summary><b>Résultats x86 complets</b> (Intel Xeon D-1531, Linux)</summary>

<br/>

*Carquet 0.9.0, Xeon D-1531 (6 cœurs / 12 threads @ 2.2GHz, AVX2), 32 Go de RAM, Ubuntu 24.04, GCC 13.3, Arrow C++ 23.0.1, PyArrow 23.0.0, fastparquet 2026.5.0, polars 2.1.0, DuckDB 1.5.0, ZSTD niveau 1, fichiers sur tmpfs, mesuré le 2026-11-22.*

#### 10 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **76.7ms** | 272ms | **3.55x** | **1.41ms** | 51.1ms | **36.3x**\* | 190.7MB |
| snappy | **90.5ms** | 527ms | **5.82x** | **35.7ms** | 94.4ms | **2.65x** | 94.9MB |
| zstd | **90.8ms** | 901ms | **9.93x** | **36.0ms** | 78.6ms | **2.18x** | 91.2MB |
| lz4 | **90.9ms** | 515ms | **5.66x** | **37.4ms** | 48.6ms | **1.30x** | 94.9MB |

</div>

<br/>

#### 1 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **8.18ms** | 27.7ms | **3.39x** | **0.19ms** | 6.10ms | **32.1x**\* | 19.1MB |
| snappy | **10.8ms** | 54.9ms | **5.10x** | **4.36ms** | 11.2ms | **2.56x** | 10.6MB |
| zstd | **13.6ms** | 91.3ms | **6.71x** | **4.70ms** | 9.13ms | **1.94x** | 9.74MB |
| lz4 | **11.1ms** | 53.5ms | **4.83x** | **4.27ms** | 5.79ms | **1.36x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **0.97ms** | 3.18ms | **3.28x** | **0.04ms** | 1.73ms | **43.2x**\* | 1.91MB |
| snappy | **3.10ms** | 5.47ms | **1.76x** | **0.68ms** | 5.14ms | **7.56x** | 1.06MB |
| zstd | **3.69ms** | 9.36ms | **2.54x** | **1.02ms** | 3.83ms | **3.75x** | 0.97MB |
| lz4 | **3.30ms** | 5.25ms | **1.59x** | **0.55ms** | 2.27ms | **4.13x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **76.7ms** | 362ms | **4.72x** | **1.41ms** | 211ms | **149.7x**\* | 190.7MB |
| snappy | **90.5ms** | 612ms | **6.77x** | **35.7ms** | 349ms | **9.80x** | 94.9MB |
| zstd | **90.8ms** | 968ms | **10.7x** | **36.0ms** | 336ms | **9.31x** | 91.2MB |
| lz4 | **90.9ms** | 607ms | **6.68x** | **37.4ms** | 245ms | **6.55x** | 94.9MB |

</div>

<br/>

#### 1 M de lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **8.18ms** | 36.6ms | **4.47x** | **0.19ms** | 8.92ms | **46.9x**\* | 19.1MB |
| snappy | **10.8ms** | 62.9ms | **5.85x** | **4.36ms** | 17.1ms | **3.92x** | 10.6MB |
| zstd | **13.6ms** | 99.5ms | **7.31x** | **4.70ms** | 18.6ms | **3.95x** | 9.74MB |
| lz4 | **11.1ms** | 62.6ms | **5.65x** | **4.27ms** | 10.4ms | **2.45x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **0.97ms** | 3.60ms | **3.71x** | **0.04ms** | 1.49ms | **37.2x**\* | 1.91MB |
| snappy | **3.10ms** | 6.33ms | **2.04x** | **0.68ms** | 5.21ms | **7.66x** | 1.06MB |
| zstd | **3.69ms** | 9.94ms | **2.69x** | **1.02ms** | 5.45ms | **5.34x** | 0.97MB |
| lz4 | **3.30ms** | 6.28ms | **1.90x** | **0.55ms** | 2.30ms | **4.18x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **76.7ms** | 222ms | **2.90x** | **1.41ms** | 105ms | **74.2x**\* | 190.7MB |
| snappy | **90.5ms** | 499ms | **5.51x** | **35.7ms** | 243ms | **6.82x** | 94.9MB |
| zstd | **90.8ms** | 929ms | **10.2x** | **36.0ms** | 410ms | **11.4x** | 91.2MB |
| lz4 | **90.9ms** | 484ms | **5.32x** | **37.4ms** | 174ms | **4.66x** | 94.9MB |

</div>

<br/>

#### 1 M de lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **8.18ms** | 32.6ms | **3.99x** | **0.19ms** | 9.84ms | **51.8x**\* | 19.1MB |
| snappy | **10.8ms** | 59.6ms | **5.54x** | **4.36ms** | 23.7ms | **5.43x** | 10.6MB |
| zstd | **13.6ms** | 99.4ms | **7.31x** | **4.70ms** | 39.8ms | **8.46x** | 9.74MB |
| lz4 | **11.1ms** | 58.9ms | **5.31x** | **4.27ms** | 16.3ms | **3.81x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **0.97ms** | 3.46ms | **3.57x** | **0.04ms** | 1.90ms | **47.5x**\* | 1.91MB |
| snappy | **3.10ms** | 6.22ms | **2.01x** | **0.68ms** | 3.35ms | **4.93x** | 1.06MB |
| zstd | **3.69ms** | 10.4ms | **2.82x** | **1.02ms** | 4.95ms | **4.85x** | 0.97MB |
| lz4 | **3.30ms** | 6.19ms | **1.88x** | **0.55ms** | 2.62ms | **4.76x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **76.7ms** | 106ms | **1.38x** | **1.41ms** | 122ms | **86.8x**\* | 190.7MB | 105.1MB |
| snappy | **90.5ms** | 110ms | **1.22x** | **35.7ms** | 88.0ms | **2.47x** | 94.9MB | 103.8MB |
| zstd | **90.8ms** | 125ms | **1.38x** | **36.0ms** | 123ms | **3.41x** | 91.2MB | 98.1MB |
| lz4 | **90.9ms** | 107ms | **1.18x** | **37.4ms** | 98.0ms | **2.62x** | 94.9MB | 104.2MB |

</div>

<br/>

#### 1 M de lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **8.18ms** | 13.3ms | **1.63x** | **0.19ms** | 8.83ms | **46.5x**\* | 19.1MB | 12.6MB |
| snappy | **10.8ms** | 14.0ms | **1.30x** | **4.36ms** | 11.1ms | **2.54x** | 10.6MB | 11.5MB |
| zstd | **13.6ms** | 16.1ms | **1.18x** | **4.70ms** | 13.9ms | **2.95x** | 9.74MB | 10.1MB |
| lz4 | **11.1ms** | 14.4ms | **1.30x** | **4.27ms** | 11.1ms | **2.60x** | 10.6MB | 11.5MB |

</div>

<br/>

#### 100 K lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **0.97ms** | 6.43ms | **6.63x** | **0.04ms** | 4.45ms | **111.2x**\* | 1.91MB | 1.26MB |
| snappy | **3.10ms** | 8.57ms | **2.76x** | **0.68ms** | 5.71ms | **8.40x** | 1.06MB | 1.15MB |
| zstd | **3.69ms** | 10.7ms | **2.89x** | **1.02ms** | 9.13ms | **8.95x** | 0.97MB | 1.01MB |
| lz4 | **3.30ms** | 8.96ms | **2.72x** | **0.55ms** | 5.93ms | **10.8x** | 1.06MB | 1.15MB |

</div>

<br/>

#### 10 M de lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **76.7ms** | 404ms | **5.27x** | **1.41ms** | 215ms | **152.5x**\* | 190.7MB | 106.9MB |
| snappy | **90.5ms** | 422ms | **4.67x** | **35.7ms** | 218ms | **6.10x** | 94.9MB | 105.7MB |
| zstd | **90.8ms** | 491ms | **5.41x** | **36.0ms** | 283ms | **7.84x** | 91.2MB | 100.2MB |
| lz4 | **90.9ms** | 436ms | **4.80x** | **37.4ms** | 229ms | **6.11x** | 94.9MB | 106.1MB |

</div>

<br/>

#### 1 M de lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **8.18ms** | 53.1ms | **6.49x** | **0.19ms** | 17.6ms | **92.4x**\* | 19.1MB | 16.1MB |
| snappy | **10.8ms** | 54.0ms | **5.02x** | **4.36ms** | 23.8ms | **5.45x** | 10.6MB | 12.4MB |
| zstd | **13.6ms** | 55.9ms | **4.11x** | **4.70ms** | 30.8ms | **6.56x** | 9.74MB | 9.92MB |
| lz4 | **11.1ms** | 57.0ms | **5.14x** | **4.27ms** | 20.2ms | **4.72x** | 10.6MB | 12.3MB |

</div>

<br/>

#### 100 K lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **0.97ms** | 7.03ms | **7.25x** | **0.04ms** | 3.21ms | **80.2x**\* | 1.91MB | 1.61MB |
| snappy | **3.10ms** | 8.53ms | **2.75x** | **0.68ms** | 4.45ms | **6.54x** | 1.06MB | 1.24MB |
| zstd | **3.69ms** | 11.5ms | **3.12x** | **1.02ms** | 5.55ms | **5.44x** | 0.97MB | 0.99MB |
| lz4 | **3.30ms** | 8.82ms | **2.67x** | **0.55ms** | 3.64ms | **6.62x** | 1.06MB | 1.23MB |

</div>

<br/>

#### Lecture croisée du même fichier (10 M de lignes)

Les deux bibliothèques lisent le **même** fichier Parquet, la comparaison la plus équitable qui soit.

<div align="center">

| Codec | Écrivain | Lecture Carquet | Lecture Arrow C++ | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **1.67ms** | 72.8ms | **43.6x**\* |
| none | Arrow | **7.49ms** | 51.7ms | **6.91x**\* |
| snappy | Carquet | **38.5ms** | 79.2ms | **2.06x** |
| snappy | Arrow | **34.2ms** | 93.7ms | **2.74x** |
| zstd | Carquet | **36.1ms** | 80.0ms | **2.22x** |
| zstd | Arrow | **45.0ms** | 76.3ms | **1.70x** |
| lz4 | Carquet | **36.0ms** | 82.2ms | **2.28x** |
| lz4 | Arrow | **27.4ms** | 49.4ms | **1.81x** |

</div>

<br/>

#### Lecture croisée du même fichier vs polars (10 M de lignes)

Chaque bibliothèque lit le fichier écrit par le benchmark de carquet et celui que polars écrit avec ses propres réglages.

<div align="center">

| Codec | Écrivain | Lecture Carquet | Lecture polars | Ratio | Fichier |
|-------|--------|-------------|-------------|-------|------|
| none | Carquet | **1.65ms** | 76.2ms | **46.2x**\* | 190.7MB |
| none | polars | **42.0ms** | 75.8ms | **1.81x** | 105.1MB |
| snappy | Carquet | **36.9ms** | 103ms | **2.78x** | 94.9MB |
| snappy | polars | **45.0ms** | 95.5ms | **2.12x** | 103.8MB |
| zstd | Carquet | **38.6ms** | 149ms | **3.85x** | 91.2MB |
| zstd | polars | **46.0ms** | 89.3ms | **1.94x** | 98.1MB |
| lz4 | Carquet | **39.8ms** | 135ms | **3.40x** | 94.9MB |
| lz4 | polars | **46.3ms** | 79.4ms | **1.71x** | 104.2MB |

</div>

\* Chemin mmap zéro-copie. fastparquet lit via pandas et ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages, ses lignes sont donc moins directement comparables.

polars utilise son propre lecteur et écrivain Parquet en Rust (`use_pyarrow=False`, son défaut) et DuckDB les siens en C++ (`COPY ... TO` depuis une table en mémoire, `read_parquet` récupéré sous forme de table Arrow), avec les mêmes données, la même taille de row group, le même niveau zstd et LZ4_RAW pour `lz4`. polars ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages ; tous deux choisissent eux-mêmes leurs encodages et encodent `id` et `category` par dictionnaire, avec ou sans codec. Sans codec, carquet conserve PLAIN, afin qu’un tel fichier soit lu directement depuis le mmap : il écrit alors 190.7 MB là où polars et DuckDB en écrivent 105 MB, et le temps d’une écriture non compressée suit le nombre d’octets qui passent par le système de fichiers. Les lectures sont comparées équitablement par le tableau de lecture croisée du même fichier.

</details>

<details>
<summary><b>Résultats ARM complets</b> (Apple M3, macOS)</summary>

<br/>

*Carquet 0.9.0, MacBook Air M3, 16 Go de RAM, macOS 26.5, Arrow C++ 26.0.1, PyArrow 26.0.1, fastparquet 2026.9.0, polars 2.1.0, DuckDB 2.0.0, ZSTD niveau 1, mesuré le 2026-11-22. Rapport brut : [`site/bench.json`](site/bench.json).*

#### 10 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **46.5ms** | 108ms | **2.32x** | **0.26ms** | 14.5ms | **55.9x**\* | 190.7MB |
| snappy | **30.4ms** | 220ms | **7.24x** | **8.19ms** | 21.9ms | **2.68x** | 94.9MB |
| zstd | **31.3ms** | 332ms | **10.6x** | **8.77ms** | 27.5ms | **3.14x** | 91.2MB |
| lz4 | **40.3ms** | 242ms | **6.01x** | **9.12ms** | 17.1ms | **1.88x** | 95.0MB |

</div>

<br/>

#### 1 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **5.60ms** | 11.6ms | **2.08x** | **0.05ms** | 1.55ms | **31.0x**\* | 19.1MB |
| snappy | **3.99ms** | 23.0ms | **5.77x** | **0.90ms** | 2.43ms | **2.70x** | 10.6MB |
| zstd | **4.23ms** | 33.3ms | **7.87x** | **1.18ms** | 3.02ms | **2.56x** | 9.74MB |
| lz4 | **3.85ms** | 24.1ms | **6.26x** | **0.91ms** | 1.82ms | **2.00x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **1.01ms** | 1.58ms | **1.56x** | **0.02ms** | 0.22ms | **11.0x**\* | 1.91MB |
| snappy | **1.20ms** | 2.47ms | **2.06x** | **0.28ms** | 0.84ms | **3.00x** | 1.06MB |
| zstd | **1.24ms** | 3.39ms | **2.73x** | **0.36ms** | 1.23ms | **3.42x** | 0.97MB |
| lz4 | **1.17ms** | 2.51ms | **2.15x** | **0.25ms** | 0.58ms | **2.32x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **46.5ms** | 156ms | **3.35x** | **0.26ms** | 25.2ms | **96.9x**\* | 190.7MB |
| snappy | **30.4ms** | 259ms | **8.52x** | **8.19ms** | 38.2ms | **4.67x** | 94.9MB |
| zstd | **31.3ms** | 383ms | **12.2x** | **8.77ms** | 40.8ms | **4.65x** | 91.2MB |
| lz4 | **40.3ms** | 283ms | **7.01x** | **9.12ms** | 28.1ms | **3.08x** | 95.0MB |

</div>

<br/>

#### 1 M de lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **5.60ms** | 16.8ms | **3.00x** | **0.05ms** | 2.68ms | **53.6x**\* | 19.1MB |
| snappy | **3.99ms** | 26.8ms | **6.72x** | **0.90ms** | 3.68ms | **4.09x** | 10.6MB |
| zstd | **4.23ms** | 38.4ms | **9.09x** | **1.18ms** | 3.78ms | **3.20x** | 9.74MB |
| lz4 | **3.85ms** | 27.8ms | **7.22x** | **0.91ms** | 2.90ms | **3.19x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs PyArrow

<div align="center">

| Codec | Écriture Carquet | Écriture PyArrow | Ratio écr. | Lecture Carquet | Lecture PyArrow | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **1.01ms** | 1.95ms | **1.93x** | **0.02ms** | 0.25ms | **12.5x**\* | 1.91MB |
| snappy | **1.20ms** | 2.80ms | **2.33x** | **0.28ms** | 0.63ms | **2.25x** | 1.06MB |
| zstd | **1.24ms** | 3.80ms | **3.06x** | **0.36ms** | 0.69ms | **1.92x** | 0.97MB |
| lz4 | **1.17ms** | 2.89ms | **2.47x** | **0.25ms** | 0.42ms | **1.68x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **46.5ms** | 56.2ms | **1.21x** | **0.26ms** | 49.0ms | **188.4x**\* | 190.7MB |
| snappy | **30.4ms** | 194ms | **6.40x** | **8.19ms** | 101ms | **12.3x** | 94.9MB |
| zstd | **31.3ms** | 322ms | **10.3x** | **8.77ms** | 183ms | **20.8x** | 91.2MB |
| lz4 | **40.3ms** | 233ms | **5.78x** | **9.12ms** | 53.6ms | **5.88x** | 95.0MB |

</div>

<br/>

#### 1 M de lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | **5.60ms** | 7.38ms | **1.32x** | **0.05ms** | 2.90ms | **58.0x**\* | 19.1MB |
| snappy | **3.99ms** | 21.3ms | **5.33x** | **0.90ms** | 9.94ms | **11.0x** | 10.6MB |
| zstd | **4.23ms** | 32.4ms | **7.65x** | **1.18ms** | 17.7ms | **15.0x** | 9.74MB |
| lz4 | **3.85ms** | 20.9ms | **5.42x** | **0.91ms** | 5.01ms | **5.51x** | 10.6MB |

</div>

<br/>

#### 100 K lignes vs fastparquet

<div align="center">

| Codec | Écriture Carquet | Écriture fastparquet | Ratio écr. | Lecture Carquet | Lecture fastparquet | Ratio lect. | Taille |
|-------|--------------|---------|---------|-------------|---------|---------|------|
| none | 1.01ms | **1.00ms** | 0.99x | **0.02ms** | 0.49ms | **24.5x**\* | 1.91MB |
| snappy | **1.20ms** | 2.26ms | **1.88x** | **0.28ms** | 1.11ms | **3.96x** | 1.06MB |
| zstd | **1.24ms** | 3.21ms | **2.59x** | **0.36ms** | 1.98ms | **5.50x** | 0.97MB |
| lz4 | **1.17ms** | 2.15ms | **1.84x** | **0.25ms** | 0.67ms | **2.68x** | 1.06MB |

</div>

<br/>

#### 10 M de lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | 46.5ms | **37.5ms** | 0.81x | **0.26ms** | 27.6ms | **106.2x**\* | 190.7MB | 105.1MB |
| snappy | **30.4ms** | 37.8ms | **1.25x** | **8.19ms** | 27.4ms | **3.35x** | 94.9MB | 103.8MB |
| zstd | **31.3ms** | 45.9ms | **1.47x** | **8.77ms** | 38.6ms | **4.40x** | 91.2MB | 98.1MB |
| lz4 | 40.3ms | **38.9ms** | 0.96x | **9.12ms** | 27.4ms | **3.00x** | 95.0MB | 104.2MB |

</div>

<br/>

#### 1 M de lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | 5.60ms | **4.16ms** | 0.74x | **0.05ms** | 2.67ms | **53.4x**\* | 19.1MB | 12.6MB |
| snappy | **3.99ms** | 4.44ms | **1.11x** | **0.90ms** | 2.87ms | **3.19x** | 10.6MB | 11.5MB |
| zstd | **4.23ms** | 5.32ms | **1.26x** | **1.18ms** | 3.82ms | **3.24x** | 9.74MB | 10.1MB |
| lz4 | **3.85ms** | 4.41ms | **1.15x** | **0.91ms** | 2.77ms | **3.04x** | 10.6MB | 11.5MB |

</div>

<br/>

#### 100 K lignes vs polars

<div align="center">

| Codec | Écriture Carquet | Écriture polars | Ratio écr. | Lecture Carquet | Lecture polars | Ratio lect. | Taille Carquet | Taille polars |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | 1.01ms | **0.68ms** | 0.67x | **0.02ms** | 0.32ms | **16.0x**\* | 1.91MB | 1.26MB |
| snappy | 1.20ms | **0.90ms** | 0.75x | **0.28ms** | 0.43ms | **1.54x** | 1.06MB | 1.15MB |
| zstd | 1.24ms | **1.04ms** | 0.84x | **0.36ms** | 1.01ms | **2.81x** | 0.97MB | 1.01MB |
| lz4 | 1.17ms | **0.93ms** | 0.79x | **0.25ms** | 0.42ms | **1.68x** | 1.06MB | 1.15MB |

</div>

<br/>

#### 10 M de lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **46.5ms** | 116ms | **2.49x** | **0.26ms** | 24.5ms | **94.4x**\* | 190.7MB | 105.7MB |
| snappy | **30.4ms** | 119ms | **3.91x** | **8.19ms** | 31.5ms | **3.84x** | 94.9MB | 104.5MB |
| zstd | **31.3ms** | 133ms | **4.25x** | **8.77ms** | 37.5ms | **4.28x** | 91.2MB | 99.1MB |
| lz4 | **40.3ms** | 119ms | **2.94x** | **9.12ms** | 26.7ms | **2.92x** | 95.0MB | 104.9MB |

</div>

<br/>

#### 1 M de lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **5.60ms** | 10.2ms | **1.82x** | **0.05ms** | 3.05ms | **61.0x**\* | 19.1MB | 16.1MB |
| snappy | **3.99ms** | 10.4ms | **2.61x** | **0.90ms** | 3.70ms | **4.11x** | 10.6MB | 12.3MB |
| zstd | **4.23ms** | 11.3ms | **2.68x** | **1.18ms** | 5.16ms | **4.37x** | 9.74MB | 9.92MB |
| lz4 | **3.85ms** | 10.6ms | **2.75x** | **0.91ms** | 3.04ms | **3.34x** | 10.6MB | 12.3MB |

</div>

<br/>

#### 100 K lignes vs DuckDB

<div align="center">

| Codec | Écriture Carquet | Écriture DuckDB | Ratio écr. | Lecture Carquet | Lecture DuckDB | Ratio lect. | Taille Carquet | Taille DuckDB |
|-------|--------------|---------|---------|-------------|---------|---------|------|------|
| none | **1.01ms** | 1.79ms | **1.77x** | **0.02ms** | 0.54ms | **27.0x**\* | 1.91MB | 1.61MB |
| snappy | **1.20ms** | 2.43ms | **2.03x** | **0.28ms** | 0.93ms | **3.32x** | 1.06MB | 1.23MB |
| zstd | **1.24ms** | 3.40ms | **2.74x** | **0.36ms** | 1.59ms | **4.42x** | 0.97MB | 0.99MB |
| lz4 | **1.17ms** | 2.75ms | **2.35x** | **0.25ms** | 0.72ms | **2.88x** | 1.06MB | 1.23MB |

</div>

<br/>

#### Lecture croisée du même fichier (10 M de lignes)

Les deux bibliothèques lisent le **même** fichier Parquet, la comparaison la plus équitable qui soit.

<div align="center">

| Codec | Écrivain | Lecture Carquet | Lecture Arrow C++ | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **0.28ms** | 16.4ms | **58.4x**\* |
| none | Arrow | **0.91ms** | 14.0ms | **15.3x**\* |
| snappy | Carquet | **8.08ms** | 17.1ms | **2.12x** |
| snappy | Arrow | **12.4ms** | 23.4ms | **1.88x** |
| zstd | Carquet | **9.40ms** | 18.3ms | **1.95x** |
| zstd | Arrow | **20.3ms** | 29.8ms | **1.47x** |
| lz4 | Carquet | **9.17ms** | 18.4ms | **2.01x** |
| lz4 | Arrow | **9.62ms** | 17.1ms | **1.78x** |

</div>

<br/>

#### Lecture croisée du même fichier vs polars (10 M de lignes)

Chaque bibliothèque lit le fichier écrit par le benchmark de carquet et celui que polars écrit avec ses propres réglages.

<div align="center">

| Codec | Écrivain | Lecture Carquet | Lecture polars | Ratio | Fichier |
|-------|--------|-------------|-------------|-------|------|
| none | Carquet | **0.29ms** | 30.0ms | **103.6x**\* | 190.7MB |
| none | polars | **9.53ms** | 23.3ms | **2.45x** | 105.1MB |
| snappy | Carquet | **8.34ms** | 31.6ms | **3.79x** | 94.9MB |
| snappy | polars | **11.0ms** | 24.7ms | **2.24x** | 103.8MB |
| zstd | Carquet | **8.89ms** | 32.8ms | **3.69x** | 91.2MB |
| zstd | polars | **17.0ms** | 34.0ms | **2.00x** | 98.1MB |
| lz4 | Carquet | **8.53ms** | 32.2ms | **3.77x** | 95.0MB |
| lz4 | polars | **11.2ms** | 24.4ms | **2.17x** | 104.2MB |

</div>

\* Chemin mmap zéro-copie. fastparquet lit via pandas et ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages, ses lignes sont donc moins directement comparables.

polars utilise son propre lecteur et écrivain Parquet en Rust (`use_pyarrow=False`, son défaut) et DuckDB les siens en C++ (`COPY ... TO` depuis une table en mémoire, `read_parquet` récupéré sous forme de table Arrow), avec les mêmes données, la même taille de row group, le même niveau zstd et LZ4_RAW pour `lz4`. polars ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages ; tous deux choisissent eux-mêmes leurs encodages et encodent `id` et `category` par dictionnaire, avec ou sans codec. Sans codec, carquet conserve PLAIN, afin qu’un tel fichier soit lu directement depuis le mmap : il écrit alors 190.7 MB là où polars et DuckDB en écrivent 105 MB, et le temps d’une écriture non compressée suit le nombre d’octets qui passent par le système de fichiers. Les lectures sont comparées équitablement par le tableau de lecture croisée du même fichier.

</details>

<details>
<summary><b>Résultats Windows complets</b> (VM Intel Haswell, MSVC)</summary>

<br/>

*Carquet 0.7.3 compilé avec MSVC 19.44 (x64 Release, OpenMP), VM OpenStack à 4 vCPU (Intel Haswell, 3.1 GHz), 14.6 Go de RAM, disque SAS virtuel, Windows Server 2025, Arrow C++ 24.0.0 (conda-forge), ZSTD niveau 1, mesuré le 2026-11-02. Les lectures à froid vont jusqu’au disque virtuel, ce qui resserre les ratios de lecture ; les écritures et le chemin zéro-copie sont limités par le CPU.*

#### 10 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. | Taille |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | **182ms** | 382ms | **2.10x** | **2.04ms** | 126.5ms | **62.0x**\* | 190.7MB |
| snappy | **175ms** | 594ms | **3.39x** | **115.6ms** | 134.9ms | **1.17x** | 125.1MB |
| zstd | **242ms** | 927ms | **3.83x** | **140.7ms** | 159.2ms | **1.13x** | 95.3MB |
| lz4 | **178ms** | 527ms | **2.96x** | **95.5ms** | 99.9ms | **1.05x** | 122.9MB |

</div>

<br/>

#### 1 M de lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **15.6ms** | 37.6ms | **2.41x** | **0.44ms** | 13.72ms | **31.2x**\* |
| snappy | **25.4ms** | 61.2ms | **2.41x** | **14.01ms** | 16.31ms | **1.16x** |
| zstd | **30.4ms** | 94.3ms | **3.10x** | **17.09ms** | 18.38ms | **1.08x** |
| lz4 | **26.4ms** | 53.8ms | **2.04x** | 13.07ms | **12.89ms** | 0.99x |

</div>

<br/>

#### 100 K lignes vs Arrow C++

<div align="center">

| Codec | Écriture Carquet | Écriture Arrow C++ | Ratio écr. | Lecture Carquet | Lecture Arrow C++ | Ratio lect. |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **3.53ms** | 3.95ms | **1.12x** | **0.16ms** | 2.13ms | **13.3x**\* |
| snappy | **5.60ms** | 6.23ms | **1.11x** | **2.79ms** | 3.92ms | **1.41x** |
| zstd | **7.11ms** | 9.79ms | **1.38x** | **2.86ms** | 4.36ms | **1.52x** |
| lz4 | **5.75ms** | 5.82ms | **1.01x** | 2.22ms | **2.15ms** | 0.97x |

</div>

<br/>

#### Lecture croisée du même fichier (10 M de lignes)

<div align="center">

| Codec | Écrivain | Lecture Carquet | Lecture Arrow C++ | Ratio |
|-------|--------|-------------|----------------|-------|
| none | Carquet | **2.65ms** | 116.9ms | **44.1x**\* |
| none | Arrow | **5.29ms** | 126.6ms | **23.9x**\* |
| snappy | Carquet | **115.2ms** | 135.0ms | **1.17x** |
| snappy | Arrow | **112.7ms** | 146.2ms | **1.30x** |
| zstd | Carquet | **137.7ms** | 150.5ms | **1.09x** |
| zstd | Arrow | **134.9ms** | 155.3ms | **1.15x** |
| lz4 | Carquet | **92.1ms** | 95.6ms | **1.04x** |
| lz4 | Arrow | **89.7ms** | 98.1ms | **1.09x** |

</div>

\* Chemin mmap zéro-copie

</details>

<details>
<summary><b>Lancer les benchmarks vous-même</b></summary>

<br/>

```bash
make benchmark                                    # Release build of the benchmark programs into build/
./build/benchmark_carquet                         # carquet alone
python3 benchmark/run_benchmark.py --skip-xlarge  # full comparison with PyArrow (+ Arrow C++ if built)
```

<div align="center">

| Drapeau / variable de `run_benchmark.py` | Effet |
|---|---|
| `--quick` | Uniquement les configurations à 10 M de lignes (celles du graphique) |
| `--skip-xlarge` | Ignorer les configurations à 100 M de lignes (~2 Go par codec, plus de 30 minutes) |
| `--fastparquet` | Comparer aussi fastparquet (`pip install fastparquet pandas`) |
| `--polars` | Comparer aussi le lecteur / écrivain Parquet Rust propre à polars (`pip install polars`) |
| `--duckdb` | Comparer aussi le lecteur / écrivain Parquet C++ propre à DuckDB (`pip install duckdb pyarrow`) |
| `--no-pyarrow` / `--no-arrow-cpp` | Ignorer PyArrow / Arrow C++ |
| `--iterations N` | Nombre d’itérations fixe (défaut : auto, 11–51) |
| `--build-dir DIR` / `--temp-dir DIR` | Emplacement des binaires / emplacement où les fichiers de test sont écrits |
| `CARQUET_BENCH_ZSTD_LEVEL=3` | Niveau ZSTD (défaut 1) |
| `CARQUET_BENCH_PHASE=write` | `benchmark_carquet` uniquement : chronométrer une seule phase (`write`, `read`, `read_nommap`), les autres ne s’exécutent qu’une fois (pour le profilage) |

</div>

Mesurez toujours un build **Release** : `make benchmark` reconfigure `build/` en Release ; un `build/` Debug resté en place fait paraître carquet plusieurs fois plus lent. `make native` + `-DCARQUET_BUILD_BENCHMARKS=ON` donne les chiffres optimisés pour la machine hôte.

fastparquet utilise les mêmes données, la même taille de row group, pas de dictionnaire et le même niveau zstd, et LZ4_RAW pour `lz4`. Il ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages, et il lit via pandas : ses chiffres sont donc moins directement comparables que ceux d’Arrow.

polars tourne avec son lecteur et son écrivain natifs (pas via PyArrow), les mêmes données, la même taille de row group, le même niveau zstd et LZ4_RAW pour `lz4`. Il ne sait écrire ni BYTE_STREAM_SPLIT ni sommes de contrôle de pages, et il choisit lui-même l’encodage par dictionnaire (sans option), ce qui rend ses fichiers plus petits sur ces données.

La comparaison Arrow C++ nécessite sa propre cible (Arrow C++ trouvé sur le système, ou désigné avec `CARQUET_ARROW_CPP_ROOT`) :

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON \
  -DCARQUET_BUILD_BENCHMARKS=ON -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON \
  [-DCARQUET_ARROW_CPP_ROOT=/path/to/arrow-prefix]
cmake --build build -j
```

Elle utilise l’API bas niveau `parquet::ParquetFileReader` (sans matérialisation en Arrow Table) avec des lecteurs de row groups parallèles.

Sous Windows (Developer PowerShell for VS), Arrow C++ peut provenir de conda-forge (`micromamba create -p C:\arrow -c conda-forge libarrow libparquet`) ; ses DLL doivent être dans le `PATH` à l’exécution du benchmark :

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_BENCHMARKS=ON `
  -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON -DCARQUET_ARROW_CPP_ROOT=C:/arrow/Library
cmake --build build
$env:PATH = "C:\arrow\Library\bin;$env:PATH"
python benchmark\run_benchmark.py --build-dir build --skip-xlarge --no-pyarrow
```

Excluez les dossiers de build et de `--temp-dir` de l’analyse antivirus en temps réel : Windows Defender analyse chaque fichier écrit par le benchmark.

Le graphique du README est tracé à partir du JSON qu’une exécution laisse dans `benchmark/`, en une variante claire et une variante sombre que GitHub choisit selon le thème du lecteur :

```bash
python3 benchmark/plot_readme_chart.py benchmark/bench_<version>_<os>_<arch>_<date>.json   # writes res/img/bench_10m_m3_{light,dark}.png (--name bench_10m_linux for the Linux figure)
```

</details>

<br/>

---

<!-- ────────────────────────────  APERÇU RAPIDE  ──────────────────────────── -->

## Trente secondes de carquet

```c
#include <carquet/carquet.h>

carquet_error_t err = CARQUET_ERROR_INIT;
carquet_schema_t* schema = carquet_schema_create(&err);
carquet_schema_add_column(schema, "id", CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

carquet_writer_t* w = carquet_writer_create("ids.parquet", schema, NULL, &err);
int64_t ids[] = {1, 2, 3};
carquet_writer_write_batch(w, 0, ids, 3, NULL, NULL);
carquet_writer_close(w);

carquet_reader_t* r = carquet_reader_open("ids.parquet", NULL, &err);
printf("%lld rows\n", (long long)carquet_reader_num_rows(r));   /* 3 rows */
carquet_reader_close(r);
carquet_schema_free(schema);
```

> **Où se situe carquet.** Arrow reste le bon choix lorsque vous avez besoin de son écosystème (noyaux de calcul, datasets, Flight, chiffrement). carquet s’adresse aux programmes C et C++, aux systèmes embarqués et aux runtimes de langages qui ont seulement besoin de lire et d’écrire du Parquet rapidement et correctement.

<br/>

---

<!-- ────────────────────────────  INSTALLATION  ──────────────────────────── -->

## Installation

Cette section installe la **bibliothèque C, ses en-têtes et la CLI `carquet`**. Les utilisateurs Python ont besoin de l’[API Python](#api-python) séparée (`pip install carquet`).

**Prérequis :** un compilateur C11 (GCC 4.9+, Clang 3.4+, MSVC 2015+) et CMake 3.16+ (ou [xmake](#depuis-les-sources-avec-xmake)). zstd, zlib, lz4 et brotli sont pris sur le système s’ils sont présents et récupérés sinon. OpenMP est optionnel (lectures et écritures parallèles).

<br/>

### Depuis les sources, avec `make`

La voie la plus simple. `make` est un fin wrapper qui appelle CMake avec les bons drapeaux, il n’y a rien d’autre à configurer :

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
make                  # optimized Release build into build/
sudo make install     # library, headers, CLI, pkg-config and CMake package files
```

Liez ensuite avec `-lcarquet`, `pkg-config --cflags --libs carquet`, ou `find_package(carquet)` dans CMake.

<br/>

### Depuis les sources, avec `xmake`

[xmake](https://xmake.io) ne nécessite rien d’autre d’installé : il télécharge des versions figées de zstd 1.5.6, zlib 1.4, lz4 1.10.0 et brotli 1.1.0 et les lie statiquement, la CLI `carquet` n’a donc aucune dépendance à l’exécution.

```bash
git clone https://github.com/Vitruves/carquet.git && cd carquet
xmake                         # Release static library + CLI into build/<plat>/<arch>/release/
xmake install -o /usr/local   # headers, library and CLI (sudo if the prefix needs it)
```

Une installation statique ne contient que `libcarquet.a` : liez donc les codecs vous-même (`-lcarquet -lzstd -lz -llz4`, plus `-lbrotlienc -lbrotlidec -lbrotlicommon` sauf si compilé avec `--brotli=n`). Configurez plutôt avec `xmake f --shared=y` pour obtenir une bibliothèque partagée autonome intégrant les codecs.

<br/>

### Dans votre projet CMake

```cmake
include(FetchContent)
FetchContent_Declare(carquet GIT_REPOSITORY https://github.com/Vitruves/carquet.git GIT_TAG main)
FetchContent_MakeAvailable(carquet)
target_link_libraries(myapp PRIVATE carquet)
```

Ou, après `make install` : `find_package(carquet REQUIRED)` puis liez `carquet::carquet`.

<br/>

---

<!-- ────────────────────────────  CONFIGURATIONS DE COMPILATION  ──────────────────────────── -->

## Configurations de compilation

### Cibles `make`

Chaque cible configure `build/` et le compile en parallèle. `make help` affiche cette liste. Chacune a un [équivalent xmake](#xmake).

<div align="center">

| Commande | Type de build | Ce que vous obtenez |
|---|---|---|
| `make` / `make release` | Release | Bibliothèque statique + CLI `carquet` (par défaut) |
| `make native` | Release | Idem, optimisé pour ce CPU avec `-march=native` (le binaire ne tourne pas sur des CPU plus anciens) |
| `make shared` | Release | Bibliothèque partagée (`.so` / `.dylib` / `.dll`) au lieu de statique |
| `make benchmark` | Release | Programmes de benchmark (`benchmark_carquet`, cibles de profilage) |
| `make debug` | Debug | Bibliothèque + CLI avec assertions, sans optimisation |
| `make test` | Debug | Suite de tests, puis son exécution avec `ctest` |
| `make dev` | Debug | Tout : tests, exemples, benchmarks, programmes d’interop |
| `make fuzz` | Debug | Cibles libFuzzer (utilisez une toolchain Clang) |
| `make install` | Release | Compile `release`, puis installe dans `PREFIX` |
| `make clean` | | Supprime le répertoire de build |

</div>

<div align="center">

| Variable | Défaut | Exemple |
|---|---|---|
| `PREFIX` | défaut de CMake (`/usr/local` sous Unix) | `make install PREFIX=$HOME/.local` |
| `BUILD` | `build` | `make test BUILD=build-debug` (garder plusieurs configurations côte à côte) |
| `JOBS` | nombre de cœurs | `make JOBS=4` |
| `CMAKE` | `cmake` | `make CMAKE=/opt/cmake/bin/cmake` |

</div>

Toutes les cibles partagent `build/` par défaut : `make test` après `make benchmark` le reconfigure donc en Debug. Utilisez `BUILD=` pour séparer une arborescence Release et une Debug.

<br/>

### Options CMake

Pour tout ce que les cibles `make` ne couvrent pas, appelez CMake directement et passez les options avec `-DOPTION=ON|OFF` :

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_BUILD_SHARED=ON -DCARQUET_ENABLE_BROTLI=OFF
cmake --build build -j
sudo cmake --install build --prefix /usr/local
```

<div align="center">

| Ce qui est compilé | Défaut | |
|---|:---:|---|
| `CARQUET_BUILD_CLI` | ON | Outil en ligne de commande `carquet` |
| `CARQUET_BUILD_SHARED` | OFF | Bibliothèque partagée au lieu de statique |
| `CARQUET_BUILD_DEV` | OFF | Active d’un coup tests, exemples, benchmarks et interop |
| `CARQUET_BUILD_TESTS` | = DEV | Suite de tests (`ctest`) |
| `CARQUET_BUILD_EXAMPLES` | = DEV | Programmes de [`examples/`](examples/) |
| `CARQUET_BUILD_BENCHMARKS` | = DEV | Programmes de benchmark et de profilage |
| `CARQUET_BUILD_ARROW_CPP_BENCHMARK` | OFF | Benchmark comparatif Arrow C++ (nécessite Arrow C++) |
| `CARQUET_BUILD_INTEROP` | = DEV | Programmes d’aller-retour C / Python |
| `CARQUET_BUILD_FUZZ` | OFF | Cibles de fuzzing |
| `CARQUET_BUILD_DOCS` | OFF | Cible `docs`, HTML Doxygen dans `build/docs/html/` |

| Dépendances | Défaut | |
|---|:---:|---|
| `CARQUET_ENABLE_BROTLI` | ON | Codec BROTLI ; OFF supprime la dépendance à libbrotli |
| `CARQUET_BUNDLE_DEPS` | OFF | Toujours récupérer et embarquer statiquement zstd / zlib / lz4 / brotli (artefacts portables, wheel Python) |
| `CARQUET_ZSTD_TARGET`, `_ZLIB_`, `_LZ4_`, `_BROTLI_TARGET` | vide | Lier une cible CMake existante au lieu de rechercher ou de récupérer (voir ci-dessous) |
| `CARQUET_ARROW_CPP_ROOT` | vide | Préfixe d’installation d’Arrow C++ pour le benchmark comparatif |
| `CMAKE_DISABLE_FIND_PACKAGE_OpenMP` | OFF | ON compile sans OpenMP (pas de `libgomp` / `vcomp140.dll` à l’exécution) |

| CPU | Défaut | |
|---|:---:|---|
| `CARQUET_NATIVE_ARCH` | OFF | `-march=native` ; le plus rapide, binaire lié à la machine hôte |
| `CARQUET_ENABLE_SSE`, `_AVX`, `_AVX2`, `_AVX512` | ON | Noyaux SIMD x86, choisis à l’exécution selon ce que le CPU prend en charge |
| `CARQUET_ENABLE_NEON` | ON | Noyaux ARM NEON |
| `CARQUET_ENABLE_SVE` | OFF | Noyaux ARM SVE (expérimental) |
| `CARQUET_MSVC_HARDENING` | ON | MSVC : `/guard:cf` et `/CETCOMPAT` |

</div>

Les noyaux SIMD sont compilés dans la bibliothèque et sélectionnés à l’exécution avec repli scalaire : un build par défaut tourne donc sur n’importe quel CPU de son architecture. Seul `CARQUET_NATIVE_ARCH` lie le binaire à la machine de compilation.

<br/>

### xmake

`xmake.lua` est une alternative complète à CMake avec les mêmes options et valeurs par défaut. Configurez une fois avec `xmake f`, puis `xmake` compile ; la configuration est conservée jusqu’au prochain `xmake f`.

<div align="center">

| Cible `make` | Équivalent xmake |
|---|---|
| `make` | `xmake` |
| `make native` | `xmake f --native_arch=y && xmake` |
| `make shared` | `xmake f --shared=y && xmake` |
| `make benchmark` | `xmake f --benchmarks=y && xmake` |
| `make debug` | `xmake f -m debug && xmake` |
| `make test` | `xmake f -m debug --tests=y && xmake && xmake test` |
| `make dev` | `xmake f -m debug --dev=y && xmake` |
| `make fuzz` | `xmake f -m debug --fuzz=y --toolchain=clang && xmake` |
| `make clean` | `xmake clean` (ou `xmake f -c` pour réinitialiser aussi la configuration) |
| `cmake --build build --target docs` | `xmake docs` (aucun drapeau de configuration nécessaire) |

</div>

Les options reflètent celles de CMake, sans le préfixe `CARQUET_` et en minuscules, sous la forme `--option=y|n`. Combinez-les librement : `xmake f -m release --shared=y --brotli=n --avx512=n`.

<div align="center">

| Option xmake | Défaut | Description |
|--------------|---------|-------------|
| `--dev` | n | Compiler tests, exemples, benchmarks et interop |
| `--tests` / `--examples` / `--benchmarks` / `--interop` | n | Compiler un groupe individuellement |
| `--arrow_cpp_benchmark` | n | Benchmark comparatif Arrow C++ (utilise Arrow depuis le dépôt xmake si disponible) |
| `--cli` | y | Compiler l’outil CLI `carquet` |
| `--shared` | n | Compiler une bibliothèque partagée plutôt que statique |
| `--openmp` | y | Lecture parallèle des colonnes via OpenMP (désactivé automatiquement si indisponible) |
| `--native_arch` | n | `-march=native` pour des performances maximales (binaire lié à la machine hôte) |
| `--sse` / `--avx` / `--avx2` / `--avx512` / `--neon` | y | Jeux d’instructions SIMD (détection automatique) |
| `--sve` | n | ARM SVE (expérimental) |
| `--brotli` | y | Codec BROTLI (libbrotli depuis le dépôt xmake) |
| `--fuzz` | n | Compiler les cibles de fuzzing (utilisez `--toolchain=clang`) ; puis `--fuzz_engine=libFuzzer\|AFL`, `--fuzz_sanitizers=y\|n` |

</div>

<details>
<summary><b>Windows</b> (MSVC, clang-cl, MinGW-w64)</summary>

<br/>

MSVC (x64 et Win32), clang-cl et MinGW-w64 GCC sont tous pris en charge, en statique comme en partagé. `CMakePresets.json` fournit les presets `msvc`, `msvc-shared`, `clang-cl` et `mingw` :

```powershell
# from a Developer PowerShell for VS (or with MinGW-w64 gcc + ninja on PATH for "mingw")
cmake --preset msvc
cmake --build --preset msvc
ctest --preset msvc
```

- Lors d’une configuration manuelle, définissez le compilateur C++ en plus du compilateur C (par exemple `CC=clang-cl CXX=clang-cl`) : le projet zstd récupéré active le C++, et CMake choisit sinon un pilote incompatible.
- Avec `-DCARQUET_BUILD_SHARED=ON`, vous obtenez `carquet.dll` et la bibliothèque d’import `carquet.lib` (`libcarquet.dll` / `libcarquet.dll.a` avec MinGW). Les consommateurs doivent définir `CARQUET_BUILD_SHARED` ; `find_package(carquet)` et `carquet.pc` l’ajoutent pour vous. Placez la DLL à côté de votre exécutable ou dans le `PATH`.
- Libérez la mémoire que carquet vous transmet, comme la sortie de `carquet_writer_get_buffer()`, avec `carquet_free()` et non `free()`. La DLL et votre programme peuvent utiliser des tas de runtime C différents (CRT Debug vs Release, `/MT`, MinGW vs MSVC).
- Les chemins de fichiers peuvent être en UTF-8 (les noms non ASCII fonctionnent). La CLI `carquet` accepte les arguments Unicode et affiche de l’UTF-8 dans la console.
- Les builds MinGW avec OpenMP nécessitent `libgomp-1.dll` et `libwinpthread-1.dll` à l’exécution. Les builds MSVC nécessitent `vcomp140.dll` (fournie par le Visual C++ Redistributable). Passez `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` pour un build sans cette dépendance.
- L’heuristique d’apprentissage automatique de Windows Defender peut mettre en quarantaine `test_encoding_roundtrip.exe` et `test_fuzz_regressions.exe` comme `Trojan:Win64/CobaltStrike.RDH!MTB` (un faux positif dû à leurs données de test binaires embarquées ; ctest les signale alors comme « Not Run »). Excluez le répertoire de build de l’analyse pour lancer la suite complète.

</details>

<details>
<summary><b>Utiliser vos propres zstd / zlib / lz4 / brotli</b> (intégrer carquet dans un projet CMake plus large)</summary>

<br/>

Lorsque carquet est intégré via `add_subdirectory()` ou `FetchContent`, il résout chaque bibliothèque de compression dans cet ordre :

1. La cible nommée par `CARQUET_ZSTD_TARGET`, `CARQUET_ZLIB_TARGET`, `CARQUET_LZ4_TARGET` ou `CARQUET_BROTLI_TARGET`, si elle est définie.
2. Une copie système (ignorée lorsque `CARQUET_BUNDLE_DEPS` est ON).
3. Une copie qu’il récupère et compile lui-même. Si votre projet compile déjà la bibliothèque sous son nom de cible habituel (`libzstd_static`, `zlibstatic`, `lz4_static`), carquet réutilise cette cible au lieu d’en récupérer une seconde copie.

Un projet qui embarque ces bibliothèques et n’a pas de copie système n’a donc besoin d’aucune configuration. Pour choisir explicitement, nommez la cible :

```cmake
set(CARQUET_ZLIB_TARGET zlibstatic)
add_subdirectory(carquet)
```

Une cible nommée est liée telle quelle, sans recherche ni récupération, et un nom qui n’est pas une cible est une erreur de configuration et non un repli silencieux. La cible peut être importée ou compilée dans l’arborescence, statique ou partagée, et doit exposer ses répertoires d’en-têtes comme exigence d’utilisation.

</details>

<details>
<summary><b>Tests, sanitizers et documentation de l’API</b></summary>

<br/>

```bash
make test                                    # Debug build + ctest
make dev && ctest --test-dir build           # everything, including examples and interop programs
cmake -B build -DCARQUET_BUILD_DOCS=ON && cmake --build build --target docs   # Doxygen → build/docs/html/
```

[CONTRIBUTING.md](CONTRIBUTING.md) décrit la barrière par pull request (tests, sanitizers, fuzzing, interop, benchmarks).

</details>

<a id="ai-coding-agents"></a>

<details>
<summary><b>Agents de codage IA</b> (un skill prêt à l’emploi pour Claude Code et Codex)</summary>

<br/>

[`.agents/skills/carquet/`](.agents/skills/carquet/) est un [Agent Skill](https://agentskills.io) qui apprend à un agent de codage à intégrer carquet dans votre projet : câblage du build (CMake, FetchContent, pkg-config, xmake), quelle API utiliser pour quoi, recettes complètes, les règles de niveaux / bitmap de nuls / durée de vie, les correspondances de types et les pièges habituels avec leurs correctifs. Copiez-le ou liez-le dans votre dépôt et l’agent le détecte automatiquement :

```bash
mkdir -p .claude/skills && cp -r path/to/carquet/.agents/skills/carquet .claude/skills/   # Claude Code
mkdir -p .agents/skills && cp -r path/to/carquet/.agents/skills/carquet .agents/skills/   # Codex CLI
ln -s ../../third_party/carquet/.agents/skills/carquet .claude/skills/carquet               # vendored: stays in sync
```

Les mêmes fichiers fonctionnent aussi comme skill personnel (`~/.claude/skills/`, `~/.codex/skills/`).

</details>

<br/>

---

<!-- ────────────────────────────  API C  ──────────────────────────── -->

## API C

Un exemple d’écriture et un exemple de lecture ci-dessous ; le [manuel dans `docs/`](docs/README.md) couvre tout le reste.

<br/>

### Écrire un fichier Parquet

```c
#include <carquet/carquet.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Define schema
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "id",    CARQUET_PHYSICAL_INT64,  NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    // Configure writer
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    // Write
    carquet_writer_t* w = carquet_writer_create("output.parquet", schema, &opts, &err);

    int64_t ids[]    = {1, 2, 3, 4, 5};
    double values[]  = {1.1, 2.2, 3.3, 4.4, 5.5};
    carquet_writer_write_batch(w, 0, ids, 5, NULL, NULL);
    carquet_writer_write_batch(w, 1, values, 5, NULL, NULL);
    carquet_writer_close(w);

    carquet_schema_free(schema);
    return 0;
}
```

<br/>

### Lire un fichier Parquet

```c
#include <carquet/carquet.h>
#include <stdio.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Open with mmap for best read performance
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;

    carquet_reader_t* r = carquet_reader_open("output.parquet", &opts, &err);
    if (!r) { printf("Error: %s\n", err.message); return 1; }

    printf("Rows: %lld, Columns: %d\n",
           (long long)carquet_reader_num_rows(r),
           carquet_reader_num_columns(r));

    // Batch reader for efficient iteration
    carquet_batch_reader_config_t cfg;
    carquet_batch_reader_config_init(&cfg);
    cfg.batch_size = 65536;

    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &cfg, &err);
    carquet_row_batch_t* batch = NULL;

    while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
        const void* data;
        const uint8_t* nulls;
        int64_t n;
        carquet_row_batch_column(batch, 0, &data, &nulls, &n);
        const int64_t* ids = (const int64_t*)data;
        // process ids[0..n-1] ...
        carquet_row_batch_free(batch);
        batch = NULL;
    }

    carquet_batch_reader_free(br);
    carquet_reader_close(r);
    return 0;
}
```

<br/>

### Autres recettes

Tout ce qui dépasse la lecture/écriture à plat se trouve dans le manuel ; chaque page renvoie vers un exemple exécutable :

<div align="center">

| Vous voulez… | Voir |
|---|---|
| Colonnes nullables, row groups, sortie en mémoire tampon | [`docs/writing.md`](docs/writing.md) |
| Listes, maps, groupes, niveaux de définition/répétition | [`docs/nested-data.md`](docs/nested-data.md) |
| Projection de colonnes, statistiques, inspection des métadonnées | [`docs/reading.md`](docs/reading.md) |
| Predicate pushdown, filtrage au niveau des pages | [`docs/reading.md`](docs/reading.md) |
| Ajouter des row groups à un fichier existant | [`docs/writing.md`](docs/writing.md#append-to-an-existing-file) |
| Compression, codecs personnalisés, réglage de l’écrivain | [`docs/writing.md`](docs/writing.md), [`docs/performance.md`](docs/performance.md) |
| mmap, zéro-copie, pré-chargement, regroupement des E/S | [`docs/performance.md`](docs/performance.md) |
| Codes d’erreur et indications de récupération | [`docs/error-handling.md`](docs/error-handling.md) |

</div>

<br/>

### Référence de l’API

L’API complète se trouve dans [`include/carquet/carquet.h`](include/carquet/carquet.h). Types principaux :

<div align="center">

| Type | Rôle |
|------|---------|
| `carquet_reader_t` | Lecteur de fichier (ouverture depuis un chemin, un `FILE*` ou un tampon mémoire) |
| `carquet_writer_t` | Écrivain de fichier |
| `carquet_batch_reader_t` | Itération par lots de haut niveau |
| `carquet_schema_t` | Définition et introspection du schéma |
| `carquet_error_t` | Informations d’erreur détaillées (code, message, emplacement dans les sources, indication de récupération) |

</div>

Le [manuel](docs/README.md) explique quelle surface utiliser et quand ; l’organisation des sources et l’architecture sont documentées dans [CONTRIBUTING.md](CONTRIBUTING.md). Une référence HTML navigable peut être générée depuis les en-têtes avec Doxygen, voir *Tests, sanitizers et documentation de l’API* sous [Configurations de compilation](#configurations-de-compilation).

<details>
<summary><b>Application d’exemple : MetricStore</b> (<code>mocklib/</code>)</summary>

<br/>

[`mocklib/`](mocklib/) est **MetricStore**, une application d’exemple complète et autonome construite sur carquet, utile à la fois comme référence d’usage réel et comme test d’intégration de bout en bout de l’API publique.

Elle modélise un magasin de télémétrie en séries temporelles (ingestion d’événements, puis introspection et requêtes) et exerce ce faisant **117 des 143 fonctions publiques de carquet (~82 %)** :

- **Côté écriture** : construction de schéma avec types logiques, réglage par colonne de l’encodage / de la compression / des filtres de Bloom, index de pages et statistiques
- **Côté lecture** : projection de colonnes, predicate pushdown, filtrage au niveau des pages, test d’appartenance Bloom, la passerelle Arrow C Data Interface, reconstruction de `LIST` imbriquées, E/S en mémoire tampon et en ajout, introspection des métadonnées

Elle lie carquet comme le ferait n’importe quel projet en aval (`find_package(carquet)` / `add_subdirectory`), fournit un test aller-retour auto-vérifié, et son binaire `write_sample` produit un fichier pour inspection externe. Voir [`mocklib/README.md`](mocklib/README.md).

</details>

<br/>

---

<!-- ────────────────────────────  API PYTHON  ──────────────────────────── -->

## API Python

`carquet` sur PyPI n’est pas un simple wrapper autour des appels C ci-dessus : c’est une **API Pythonique, dans le style de `pyarrow.parquet`** (`read_table`, `ParquetFile`, `ParquetWriter`, `filters` DNF) construite sur le même moteur. La wheel embarque la bibliothèque C avec tous les codecs compilés ; les données passent vers Python via l’Arrow C Data Interface, si bien que le décodage, la décompression et les E/S s’exécutent en C sans le GIL. **Elle n’a aucune dépendance** : les lectures renvoient une `carquet.Table` qui se convertit seule en valeurs Python, NumPy et pandas, et que polars, DuckDB et PyArrow prennent sans copie.

```bash
pip install carquet              # no dependencies: no PyArrow, no NumPy needed
pip install "carquet[pandas]"    # with pandas + NumPy, for read_pandas() / to_pandas()
```

```python
import carquet as cq

table = cq.read_table("data.parquet", columns=["id", "price"], filters=[("price", ">", 100)])
df    = cq.read_pandas("data.parquet")                          # pandas.DataFrame, index restored
cq.write_table(table, "out.parquet", compression="zstd")        # also DataFrame, polars, dict, batches

with cq.ParquetFile("data.parquet") as pf:
    for batch in pf.iter_batches(batch_size=50_000, columns=["id"]):
        ...
    pf.metadata.row_group(0).column("price").statistics       # min / max / null_count
```

<div align="center">

| Vous voulez… | Utilisez |
|---|---|
| Lire un fichier, un répertoire, des octets ou un objet fichier | `cq.read_table`, `cq.read_pandas` |
| Parcourir de gros fichiers en flux | `ParquetFile.iter_batches`, `cq.iter_batches` |
| Élaguer avec les statistiques, les index de pages, les filtres de Bloom | `filters=` (DNF, résultats exacts), `ParquetFile.bloom_filter` |
| Écrire des tables, DataFrames, polars, résultats DuckDB | `cq.write_table`, `cq.ParquetWriter` (incrémental, `append=True`) |
| Passer les données à pandas, NumPy, polars, DuckDB, PyArrow | `t.to_pandas()`, `t.column("x").to_numpy()`, `polars.DataFrame(t)`, `pyarrow.table(t)` |
| Inspecter un fichier sans lire ses données | `cq.read_metadata`, `cq.read_parquet_schema`, `cq.info`, `cq.validate` |

</div>

Les types font l’aller-retour à l’identique avec `pyarrow.parquet` (fuseaux horaires, catégoriels, décimaux, types large, données imbriquées). La référence complète se trouve dans [`python/README.md`](python/README.md).

<br/>

---

<!-- ────────────────────────────  OUTIL CLI  ──────────────────────────── -->

## Outil en ligne de commande

Carquet est livré avec un outil en ligne de commande pour inspecter les fichiers Parquet et générer du code de lecture C. Il est compilé et installé par défaut avec la bibliothèque.

```
Commands:
  schema     Print file schema
  info       Print detailed file metadata
  head       Print first N rows
  tail       Print last N rows
  cat        Print rows with slicing/column/row filtering
  count      Print total row count
  columns    List column names (one per line)
  stat       Print column statistics
  validate   Verify file integrity
  sample     Print N random rows
  export     Write rows to stdout as CSV
  codegen    Generate C reader code
```

Usage typique :

```bash
carquet schema data.parquet
carquet head -n 20 data.parquet
carquet stat data.parquet
carquet validate data.parquet
```

<br/>

### Filtrage des lignes

`cat`, `count`, `head`, `tail`, `sample` et `export` acceptent `-p / --filter EXPR` pour pousser un prédicat de ligne jusqu’au niveau des pages : seules les pages dont les min/max de l’index de colonnes peuvent satisfaire le prédicat sont décompressées.

```bash
carquet cat -p "price > 100 AND status = 'active'" data.parquet
carquet count --filter "id >= 1000" data.parquet
carquet export --filter "ts IS NOT NULL" -c id,ts data.parquet
```

La grammaire est `column OP value [AND column OP value]...` avec `OP` ∈ {`=`, `==`, `!=`, `<>`, `<`, `<=`, `>`, `>=`}, plus `column IS NULL` / `column IS NOT NULL`.

> Le filtrage nécessite que le fichier possède un index de pages (`write_page_index = true`).

<br/>

### Génération de code

Générez un lecteur C complet et compilable à partir du schéma de n’importe quel fichier Parquet :

```bash
carquet codegen -f data.parquet -o reader.c
# Generated: reader.c
# Compile:   clang -o reader reader.c -I.../include -L.../build -lcarquet ...

./reader                    # reads data.parquet (embedded as default)
./reader other.parquet      # override with a different file
```

<details>
<summary><b>Options de codegen</b></summary>

<br/>

<div align="center">

| Drapeau | Description |
|------|-------------|
| `-f`, `--file FILE` | Fichier Parquet dont inspecter le schéma |
| `-o`, `--output FILE` | Fichier source de sortie (défaut : stdout) |
| `--mmap` | Utiliser des E/S en mémoire mappée dans le code généré |
| `--skeleton` | Générer un `process_batch` vide pour votre propre logique |
| `-c`, `--columns COLS` | Filtre de colonnes séparées par des virgules |
| `-b`, `--batch-size N` | Taille des lots (défaut : 1024) |

</div>

</details>

<br/>

---

<!-- ────────────────────────────  FONCTIONNALITÉS  ──────────────────────────── -->

## Fonctionnalités Parquet prises en charge

Ce que la bibliothèque implémente aujourd’hui, par domaine.

<br/>

### Format

<div align="center">

| Fonctionnalité | État |
|---------|--------|
| Types physiques | ✓  Les 8 (de BOOLEAN à FIXED_LEN_BYTE_ARRAY) |
| Types logiques | ✓  STRING, DATE, TIME, TIMESTAMP, DECIMAL, UUID, JSON, INTERVAL, FLOAT16, VARIANT, GEOMETRY, GEOGRAPHY |
| Encodages | ✓  PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT (lecture + écriture) |
| Versions de Data Page | ✓  V1 (défaut) et V2 (lecture + écriture) |
| Compression | ✓  UNCOMPRESSED, SNAPPY, GZIP, LZ4 (tramage Hadoop, codec 5), LZ4_RAW (codec 7), ZSTD, BROTLI (`CARQUET_ENABLE_BROTLI`, ON par défaut) ; LZO via `carquet_register_codec()` |
| Schémas imbriqués | ✓  Groupes, listes, maps avec niveaux de définition/répétition ; découpage automatique des LIST/MAP à un niveau en écriture (`carquet_writer_write_list_column`) et reconstruction des List en lecture ; émission d’`ARROW:schema` imbriqué |
| Chiffrement | ✗ Non pris en charge ; un fichier à pied de page chiffré (`PARE`) est signalé comme `CARQUET_ERROR_NOT_IMPLEMENTED` avec un message explicite |

</div>

<details>
<summary><b>Indexation &amp; élagage</b></summary>

<br/>

<div align="center">

| Fonctionnalité | État |
|---------|--------|
| Filtres de Bloom | ✓  Lecture, écriture et interrogation (`carquet_bloom_filter_check_*`) ; consultés automatiquement pour l’élagage des row groups |
| Index de pages | ✓  Index de colonnes + index d’offsets (lecture + écriture + accès aux statistiques par page) |
| Statistiques | ✓  Min/max/nombre de nuls par column chunk ; `distinct_count` exact pour les colonnes à dictionnaire ; SizeStatistics Parquet 2.9 (octets de byte-array non encodés + histogrammes de niveaux) |
| Predicate pushdown | ✓  Élagage automatique des row groups via statistiques + filtres de Bloom (aucun callback nécessaire) ; filtrage au niveau des pages via l’index de colonnes (`carquet_batch_reader_set_page_filter`) |
| Statistiques géospatiales | ✓  Boîte englobante et types de géométrie par colonne pour GEOMETRY/GEOGRAPHY (`carquet_reader_geospatial_statistics`) |
| Projection de colonnes | ✓  Lire uniquement les colonnes sélectionnées |
| Fenêtre de lignes | ✓  Lire une tranche `[offset, limit)` de lignes en sautant des pages entières (`carquet_batch_reader_set_row_range`) |

</div>

</details>

<details>
<summary><b>E/S &amp; extensibilité</b></summary>

<br/>

<div align="center">

| Fonctionnalité | État |
|---------|--------|
| Ajout | ✓  Ajouter des row groups à un fichier existant (`carquet_writer_open_append`) |
| Codecs personnalisés | ✓  Enregistrer une implémentation compress/decompress personnalisée par emplacement de codec (`carquet_register_codec`) |
| Métadonnées clé-valeur | ✓  Lire et écrire des métadonnées arbitraires dans le pied de page |
| Métadonnées par champ | ✓  `Field.custom_metadata` Arrow (libellés/descriptions de variables) via `ARROW:schema` (lecture + écriture) |
| Options par colonne | ✓  Encodage, compression, statistiques et filtre de Bloom par colonne |
| Écrivain en mémoire | ✓  Écrire du Parquet dans un tampon en mémoire |
| CRC32 | ✓  Vérification au niveau des pages (accélérée matériellement sur ARM) |
| E/S en mémoire mappée | ✓  Lectures zéro-copie pour les données PLAIN non compressées |
| Lectures préservant le dictionnaire | ✓  Renvoyer les indices RLE + le dictionnaire au lieu des valeurs matérialisées (`preserve_dictionaries`, `carquet_row_batch_column_dictionary`) |
| Regroupement des E/S | ✓  Pré-chargement des lectures multi-colonnes en une seule E/S |
| Pied de page spéculatif | ✓  Ouverture de fichier en une seule E/S pour la plupart des fichiers |
| Lectures parallèles OpenMP | ✓  Lorsque disponible |
| Écritures parallèles | ✓  Encodage/compression parallèles au niveau des pages entre colonnes (OpenMP) + E/S des row groups en arrière-plan (`async_io`) |

</div>

</details>

<br/>

---

<!-- ────────────────────────────  INTEROPÉRABILITÉ  ──────────────────────────── -->

## Interopérabilité

Les fichiers Carquet sont entièrement compatibles avec PyArrow, DuckDB, Spark et tout lecteur Parquet.

```python
import pyarrow.parquet as pq
table = pq.read_table("carquet_output.parquet")  # just works
```

```sql
-- DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

<details>
<summary><b>Suite de tests d’interopérabilité</b></summary>

<br/>

Allers-retours bidirectionnels (C écrit / Python lit, et inversement) :

```bash
cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build
python3 interop/run_interop.py
```

Les gros fichiers Snappy multi-row-group (nombreux row groups, larges colonnes `BYTE_ARRAY`, offsets de pages au-delà de 2 Gio) ont leur propre générateur, qui construit le fichier avec PyArrow et vérifie que carquet décode chaque page :

```bash
CARQUET_BIN=build/carquet python3 interop/snappy_multi_rowgroup.py --rows 150000 --row-groups 92
```

</details>

<br/>

---

<!-- ────────────────────────────  CONTRIBUER & LICENCE  ──────────────────────────── -->

## Contribuer, citer, licence

- **Contribuer** : [CONTRIBUTING.md](CONTRIBUTING.md) couvre l’installation, le style de code et la barrière par pull request (tests, sanitizers, fuzzing, interop, benchmarks). Les notes de version sont dans [CHANGELOG.md](CHANGELOG.md).
- **Citer** : un fichier [`CITATION.cff`](CITATION.cff) est inclus ; le bouton « Cite this repository » de GitHub le formate en BibTeX ou APA.
- **Licence** : MIT, voir [LICENSE](LICENSE).
