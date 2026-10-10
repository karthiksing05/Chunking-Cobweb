## Several seeds (13, 17, 23)

Each seed draws its own split of learned and held-out characters (the seed of `run_characters.py`) and its own shuffles. Mean over the seeds with the value, then min–max; n = number of seeds.

| Quantity | seed 13 | seed 17 | seed 23 | mean | min–max | n |
|---|---|---|---|---|---|---|
| held-out · token 1-gram · real | 44.05 | 44.39 | 45.24 | 44.56 | 44.05–45.24 | 3 |
| held-out · token 1-gram · shuffled | 43.49 | 44.84 | 44.80 | 44.38 | 43.49–44.84 | 3 |
| held-out · token 1-gram · slot_shuffled | 43.82 | 44.50 | 45.21 | 44.51 | 43.82–45.21 | 3 |
| held-out · token 2-gram · real | 31.20 | 31.30 | 32.09 | 31.53 | 31.20–32.09 | 3 |
| held-out · token 2-gram · shuffled | 43.06 | 44.05 | 44.16 | 43.76 | 43.06–44.16 | 3 |
| held-out · token 2-gram · slot_shuffled | 36.85 | 36.97 | 38.10 | 37.31 | 36.85–38.10 | 3 |
| gain lost when shuffled · token 2-gram | 12.42 | 12.31 | 12.52 | 12.42 | 12.31–12.52 | 3 |
| gain lost when slot_shuffled · token 2-gram | 5.88 | 5.56 | 6.05 | 5.83 | 5.56–6.05 | 3 |
| held-out · token 3-gram · real | 29.03 | 29.50 | 30.52 | 29.69 | 29.03–30.52 | 3 |
| held-out · token 3-gram · shuffled | 49.60 | 50.28 | 50.44 | 50.10 | 49.60–50.44 | 3 |
| held-out · token 3-gram · slot_shuffled | 42.65 | 43.05 | 43.94 | 43.21 | 42.65–43.94 | 3 |
| gain lost when shuffled · token 3-gram | 21.13 | 20.32 | 20.36 | 20.60 | 20.32–21.13 | 3 |
| gain lost when slot_shuffled · token 3-gram | 13.85 | 13.44 | 13.45 | 13.58 | 13.44–13.85 | 3 |
| held-out · xz -9e · real | 35.01 | 35.58 | 36.22 | 35.61 | 35.01–36.22 | 3 |
| held-out · xz -9e · shuffled | 56.58 | 58.30 | 57.92 | 57.60 | 56.58–58.30 | 3 |
| held-out · xz -9e · slot_shuffled | 49.47 | 49.98 | 51.26 | 50.24 | 49.47–51.26 | 3 |
| gain lost when shuffled · xz -9e | 22.13 | 22.27 | 22.14 | 22.18 | 22.13–22.27 | 3 |
| gain lost when slot_shuffled · xz -9e | 14.70 | 14.29 | 15.08 | 14.69 | 14.29–15.08 | 3 |
| held-out · GRU 8 · real | 29.40 | 29.09 | 29.91 | 29.47 | 29.09–29.91 | 3 |
| held-out · GRU 8 · shuffled | 37.11 | 37.56 | 37.78 | 37.48 | 37.11–37.78 | 3 |
| held-out · GRU 8 · slot_shuffled | 32.29 | 33.19 | 33.45 | 32.98 | 32.29–33.45 | 3 |
| gain lost when shuffled · GRU 8 | 8.27 | 8.02 | 8.32 | 8.20 | 8.02–8.32 | 3 |
| gain lost when slot_shuffled · GRU 8 | 3.13 | 3.99 | 3.57 | 3.56 | 3.13–3.99 | 3 |
| held-out · GRU 16 · real | 26.64 | 26.53 | 27.58 | 26.92 | 26.53–27.58 | 3 |
| held-out · GRU 16 · shuffled | 36.40 | 37.62 | 37.37 | 37.13 | 36.40–37.62 | 3 |
| held-out · GRU 16 · slot_shuffled | 31.59 | 31.84 | 33.04 | 32.16 | 31.59–33.04 | 3 |
| gain lost when shuffled · GRU 16 | 10.32 | 10.64 | 10.23 | 10.40 | 10.23–10.64 | 3 |
| gain lost when slot_shuffled · GRU 16 | 5.18 | 5.19 | 5.49 | 5.29 | 5.18–5.49 | 3 |
| held-out · GRU 32 · real | 24.24 | 24.57 | 25.42 | 24.74 | 24.24–25.42 | 3 |
| held-out · GRU 32 · shuffled | 36.26 | 37.50 | 37.38 | 37.05 | 36.26–37.50 | 3 |
| held-out · GRU 32 · slot_shuffled | 31.04 | 31.60 | 32.35 | 31.66 | 31.04–32.35 | 3 |
| gain lost when shuffled · GRU 32 | 12.58 | 12.48 | 12.41 | 12.49 | 12.41–12.58 | 3 |
| gain lost when slot_shuffled · GRU 32 | 7.02 | 6.92 | 6.96 | 6.97 | 6.92–7.02 | 3 |
| held-out · GRU 64 · real | 23.08 | 23.64 | 24.76 | 23.83 | 23.08–24.76 | 3 |
| held-out · GRU 64 · shuffled | 36.31 | 37.63 | 37.38 | 37.11 | 36.31–37.63 | 3 |
| held-out · GRU 64 · slot_shuffled | 31.10 | 31.24 | 32.17 | 31.50 | 31.10–32.17 | 3 |
| gain lost when shuffled · GRU 64 | 13.79 | 13.54 | 13.07 | 13.46 | 13.07–13.79 | 3 |
| gain lost when slot_shuffled · GRU 64 | 8.25 | 7.49 | 7.44 | 7.73 | 7.44–8.25 | 3 |
| held-out · GRU 128 · real | 22.70 | 22.91 | 23.61 | 23.07 | 22.70–23.61 | 3 |
| held-out · GRU 128 · shuffled | 36.52 | 37.82 | 37.46 | 37.27 | 36.52–37.82 | 3 |
| held-out · GRU 128 · slot_shuffled | 31.07 | 31.30 | 32.08 | 31.49 | 31.07–32.08 | 3 |
| gain lost when shuffled · GRU 128 | 14.38 | 14.46 | 14.29 | 14.38 | 14.29–14.46 | 3 |
| gain lost when slot_shuffled · GRU 128 | 8.60 | 8.28 | 8.51 | 8.46 | 8.28–8.60 | 3 |
| held-out · TRELLIS relational · real | 30.27 | 30.56 | 31.74 | 30.86 | 30.27–31.74 | 3 |
| held-out · TRELLIS relational · shuffled | 35.78 | 37.12 | 36.85 | 36.58 | 35.78–37.12 | 3 |
| held-out · TRELLIS relational · slot_shuffled | 31.18 | 31.35 | 32.21 | 31.58 | 31.18–32.21 | 3 |
| gain lost when shuffled · TRELLIS relational | 6.08 | 6.10 | 5.55 | 5.91 | 5.55–6.10 | 3 |
| gain lost when slot_shuffled · TRELLIS relational | 1.15 | 0.67 | 0.50 | 0.77 | 0.50–1.15 | 3 |
| held-out · TRELLIS supervised · real | 24.90 | 24.72 | 25.69 | 25.10 | 24.72–25.69 | 3 |
| held-out · TRELLIS supervised · shuffled | 36.64 | 37.64 | 37.56 | 37.28 | 36.64–37.64 | 3 |
| held-out · TRELLIS supervised · slot_shuffled | 33.97 | 32.85 | 33.67 | 33.50 | 32.85–33.97 | 3 |
| gain lost when shuffled · TRELLIS supervised | 12.30 | 12.47 | 12.32 | 12.37 | 12.30–12.47 | 3 |
| gain lost when slot_shuffled · TRELLIS supervised | 9.31 | 8.02 | 8.02 | 8.45 | 8.02–9.31 | 3 |
| held-out · TRELLIS unsupervised · real | 25.64 | 26.34 | 26.82 | 26.26 | 25.64–26.82 | 3 |
| held-out · TRELLIS unsupervised · shuffled | 39.40 | 40.73 | 40.46 | 40.20 | 39.40–40.73 | 3 |
| held-out · TRELLIS unsupervised · slot_shuffled | 35.15 | 34.41 | 35.85 | 35.14 | 34.41–35.85 | 3 |
| gain lost when shuffled · TRELLIS unsupervised | 14.32 | 13.94 | 14.09 | 14.12 | 13.94–14.32 | 3 |
| gain lost when slot_shuffled · TRELLIS unsupervised | 9.74 | 7.96 | 9.07 | 8.92 | 7.96–9.74 | 3 |
| total · token 2-gram (adaptive code) · real | 34.88 | 34.44 | 34.86 | 34.73 | 34.44–34.88 | 3 |
| total · token 2-gram (adaptive code) · shuffled | 46.44 | 46.04 | 46.16 | 46.21 | 46.04–46.44 | 3 |
| total · token 2-gram (adaptive code) · slot_shuffled | 40.05 | 39.73 | 39.94 | 39.91 | 39.73–40.05 | 3 |
| total · token 3-gram (adaptive code) · real | 35.18 | 34.51 | 35.09 | 34.93 | 34.51–35.18 | 3 |
| total · token 3-gram (adaptive code) · shuffled | 53.08 | 52.65 | 52.84 | 52.85 | 52.65–53.08 | 3 |
| total · token 3-gram (adaptive code) · slot_shuffled | 46.94 | 46.58 | 46.97 | 46.83 | 46.58–46.97 | 3 |
| total · xz -9e · real | 42.91 | 42.43 | 42.96 | 42.77 | 42.43–42.96 | 3 |
| total · xz -9e · shuffled | 62.86 | 62.50 | 62.50 | 62.62 | 62.50–62.86 | 3 |
| total · xz -9e · slot_shuffled | 55.71 | 55.28 | 55.60 | 55.53 | 55.28–55.71 | 3 |
| total · GRU 8 (8-bit) · real | 46.54 | 45.82 | 46.47 | 46.28 | 45.82–46.54 | 3 |
| total · GRU 8 (8-bit) · shuffled | 56.93 | 55.19 | 56.04 | 56.06 | 55.19–56.93 | 3 |
| total · GRU 8 (8-bit) · slot_shuffled | 50.58 | 50.62 | 50.48 | 50.56 | 50.48–50.62 | 3 |
| total · GRU 8 (entropy-coded, best bit depth) · real | 39.04 | 39.96 | 38.38 | 39.13 | 38.38–39.96 | 3 |
| total · GRU 8 (entropy-coded, best bit depth) · shuffled | 45.82 | 44.81 | 44.56 | 45.06 | 44.56–45.82 | 3 |
| total · GRU 8 (entropy-coded, best bit depth) · slot_shuffled | 41.65 | 41.56 | 41.33 | 41.51 | 41.33–41.65 | 3 |
| total · GRU 16 (8-bit) · real | 64.37 | 62.95 | 63.33 | 63.55 | 62.95–64.37 | 3 |
| total · GRU 16 (8-bit) · shuffled | 77.58 | 76.01 | 76.88 | 76.82 | 76.01–77.58 | 3 |
| total · GRU 16 (8-bit) · slot_shuffled | 71.16 | 69.79 | 71.10 | 70.68 | 69.79–71.16 | 3 |
| total · GRU 16 (entropy-coded, best bit depth) · real | 43.03 | 41.71 | 42.54 | 42.43 | 41.71–43.03 | 3 |
| total · GRU 16 (entropy-coded, best bit depth) · shuffled | 49.81 | 48.16 | 48.25 | 48.74 | 48.16–49.81 | 3 |
| total · GRU 16 (entropy-coded, best bit depth) · slot_shuffled | 46.47 | 48.70 | 48.93 | 48.03 | 46.47–48.93 | 3 |
| total · GRU 32 (8-bit) · real | 113.40 | 110.15 | 112.09 | 111.88 | 110.15–113.40 | 3 |
| total · GRU 32 (8-bit) · shuffled | 129.82 | 126.44 | 128.29 | 128.18 | 126.44–129.82 | 3 |
| total · GRU 32 (8-bit) · slot_shuffled | 122.33 | 120.51 | 121.61 | 121.48 | 120.51–122.33 | 3 |
| total · GRU 32 (entropy-coded, best bit depth) · real | 52.51 | 50.10 | 54.36 | 52.33 | 50.10–54.36 | 3 |
| total · GRU 32 (entropy-coded, best bit depth) · shuffled | 58.28 | 57.86 | 58.67 | 58.27 | 57.86–58.67 | 3 |
| total · GRU 32 (entropy-coded, best bit depth) · slot_shuffled | 57.20 | 55.61 | 54.87 | 55.89 | 54.87–57.20 | 3 |
| total · GRU 64 (8-bit) · real | 253.73 | 245.71 | 248.83 | 249.42 | 245.71–253.73 | 3 |
| total · GRU 64 (8-bit) · shuffled | 271.41 | 264.98 | 268.93 | 268.44 | 264.98–271.41 | 3 |
| total · GRU 64 (8-bit) · slot_shuffled | 263.12 | 258.12 | 261.67 | 260.97 | 258.12–263.12 | 3 |
| total · GRU 64 (entropy-coded, best bit depth) · real | 73.72 | 68.80 | 73.98 | 72.17 | 68.80–73.98 | 3 |
| total · GRU 64 (entropy-coded, best bit depth) · shuffled | 78.21 | 80.80 | 73.50 | 77.50 | 73.50–80.80 | 3 |
| total · GRU 64 (entropy-coded, best bit depth) · slot_shuffled | 84.44 | 71.93 | 66.04 | 74.14 | 66.04–84.44 | 3 |
| total · GRU 128 (8-bit) · real | 683.05 | 667.35 | 674.02 | 674.81 | 667.35–683.05 | 3 |
| total · GRU 128 (8-bit) · shuffled | 702.36 | 689.74 | 697.45 | 696.52 | 689.74–702.36 | 3 |
| total · GRU 128 (8-bit) · slot_shuffled | 692.19 | 682.61 | 687.20 | 687.33 | 682.61–692.19 | 3 |
| total · GRU 128 (entropy-coded, best bit depth) · real | 120.73 | 118.54 | 117.39 | 118.89 | 117.39–120.73 | 3 |
| total · GRU 128 (entropy-coded, best bit depth) · shuffled | 127.98 | 124.92 | 130.06 | 127.65 | 124.92–130.06 | 3 |
| total · GRU 128 (entropy-coded, best bit depth) · slot_shuffled | 119.55 | 117.82 | 117.87 | 118.42 | 117.82–119.55 | 3 |
| total · TRELLIS relational · real | 34.51 | 34.49 | 34.41 | 34.47 | 34.41–34.51 | 3 |
| total · TRELLIS relational · shuffled | 38.44 | 38.03 | 38.10 | 38.19 | 38.03–38.44 | 3 |
| total · TRELLIS relational · slot_shuffled | 35.15 | 34.70 | 34.66 | 34.83 | 34.66–35.15 | 3 |
| total · TRELLIS supervised · real | 31.70 | 31.02 | 31.55 | 31.43 | 31.02–31.70 | 3 |
| total · TRELLIS supervised · shuffled | 39.54 | 39.12 | 39.27 | 39.31 | 39.12–39.54 | 3 |
| total · TRELLIS supervised · slot_shuffled | 36.93 | 36.12 | 35.98 | 36.34 | 35.98–36.93 | 3 |
| total · TRELLIS unsupervised · real | 31.99 | 32.24 | 32.32 | 32.18 | 31.99–32.32 | 3 |
| total · TRELLIS unsupervised · shuffled | 42.19 | 41.88 | 41.90 | 41.99 | 41.88–42.19 | 3 |
| total · TRELLIS unsupervised · slot_shuffled | 39.44 | 37.78 | 38.90 | 38.71 | 37.78–39.44 | 3 |
| validity: modal yield a component · relational · real | 87.9% | 87.1% | 89.7% | 88.2% | 87.1%–89.7% | 3 |
| validity: mean purity · relational · real | 0.73 | 0.65 | 0.58 | 0.65 | 0.58–0.73 | 3 |
| validity: specific reused unit · relational · real | 60.0% | 48.8% | 49.3% | 52.7% | 48.8%–60.0% | 3 |
| validity: chunk types · relational · real | 280 | 201 | 136 | 206 | 136–280 | 3 |
| recall@50, chunk type purity >= 0.5 · relational | 30.0% | 44.0% | 22.0% | 32.0% | 22.0%–44.0% | 3 |
| recall@50, chunk type any purity · relational | 46.0% | 64.0% | 42.0% | 50.7% | 42.0%–64.0% | 3 |
| recall@100, chunk type purity >= 0.5 · relational | 27.0% | 30.0% | 15.0% | 24.0% | 15.0%–30.0% | 3 |
| recall@100, chunk type any purity · relational | 41.0% | 46.0% | 31.0% | 39.3% | 31.0%–46.0% | 3 |
| recall@200, chunk type purity >= 0.5 · relational | 21.0% | 21.5% | 11.5% | 18.0% | 11.5%–21.5% | 3 |
| recall@200, chunk type any purity · relational | 30.5% | 32.5% | 22.0% | 28.3% | 22.0%–32.5% | 3 |
| validity: modal yield a component · relational · shuffled | 4.3% | 2.1% | 2.6% | 3.0% | 2.1%–4.3% | 3 |
| validity: mean purity · relational · shuffled | 0.17 | 0.21 | 0.23 | 0.20 | 0.17–0.23 | 3 |
| validity: specific reused unit · relational · shuffled | 0.0% | 0.0% | 0.0% | 0.0% | 0.0%–0.0% | 3 |
| validity: chunk types · relational · shuffled | 92 | 96 | 114 | 101 | 92–114 | 3 |
| validity: modal yield a component · relational · slot_shuffled | 20.2% | 19.6% | 20.6% | 20.1% | 19.6%–20.6% | 3 |
| validity: mean purity · relational · slot_shuffled | 0.45 | 0.43 | 0.35 | 0.41 | 0.35–0.45 | 3 |
| validity: specific reused unit · relational · slot_shuffled | 9.7% | 7.1% | 6.5% | 7.8% | 6.5%–9.7% | 3 |
| validity: chunk types · relational · slot_shuffled | 124 | 112 | 107 | 114 | 107–124 | 3 |
| validity: modal yield a component · supervised · real | 30.3% | 19.5% | 23.0% | 24.3% | 19.5%–30.3% | 3 |
| validity: mean purity · supervised · real | 0.48 | 0.50 | 0.60 | 0.53 | 0.48–0.60 | 3 |
| validity: specific reused unit · supervised · real | 14.6% | 6.9% | 13.3% | 11.6% | 6.9%–14.6% | 3 |
| validity: chunk types · supervised · real | 89 | 87 | 113 | 96 | 87–113 | 3 |
| recall@50, chunk type purity >= 0.5 · supervised | 8.0% | 6.0% | 8.0% | 7.3% | 6.0%–8.0% | 3 |
| recall@50, chunk type any purity · supervised | 20.0% | 12.0% | 20.0% | 17.3% | 12.0%–20.0% | 3 |
| recall@100, chunk type purity >= 0.5 · supervised | 5.0% | 3.0% | 6.0% | 4.7% | 3.0%–6.0% | 3 |
| recall@100, chunk type any purity · supervised | 13.0% | 6.0% | 13.0% | 10.7% | 6.0%–13.0% | 3 |
| recall@200, chunk type purity >= 0.5 · supervised | 4.0% | 2.5% | 3.5% | 3.3% | 2.5%–4.0% | 3 |
| recall@200, chunk type any purity · supervised | 8.5% | 4.5% | 7.0% | 6.7% | 4.5%–8.5% | 3 |
| validity: modal yield a component · supervised · shuffled | 12.1% | 4.7% | 4.3% | 7.0% | 4.3%–12.1% | 3 |
| validity: mean purity · supervised · shuffled | 0.23 | 0.38 | 0.28 | 0.30 | 0.23–0.38 | 3 |
| validity: specific reused unit · supervised · shuffled | 0.0% | 0.0% | 0.0% | 0.0% | 0.0%–0.0% | 3 |
| validity: chunk types · supervised · shuffled | 33 | 43 | 47 | 41 | 33–47 | 3 |
| validity: modal yield a component · supervised · slot_shuffled | 9.1% | 9.5% | 11.5% | 10.1% | 9.1%–11.5% | 3 |
| validity: mean purity · supervised · slot_shuffled | 0.27 | 0.37 | 0.24 | 0.29 | 0.24–0.37 | 3 |
| validity: specific reused unit · supervised · slot_shuffled | 0.0% | 4.8% | 0.0% | 1.6% | 0.0%–4.8% | 3 |
| validity: chunk types · supervised · slot_shuffled | 44 | 63 | 52 | 53 | 44–63 | 3 |
| validity: modal yield a component · unsupervised · real | 44.0% | 38.1% | 36.8% | 39.6% | 36.8%–44.0% | 3 |
| validity: mean purity · unsupervised · real | 0.43 | 0.41 | 0.48 | 0.44 | 0.41–0.48 | 3 |
| validity: specific reused unit · unsupervised · real | 14.0% | 9.5% | 7.0% | 10.2% | 7.0%–14.0% | 3 |
| validity: chunk types · unsupervised · real | 50 | 42 | 57 | 50 | 42–57 | 3 |
| recall@50, chunk type purity >= 0.5 · unsupervised | 12.0% | 4.0% | 4.0% | 6.7% | 4.0%–12.0% | 3 |
| recall@50, chunk type any purity · unsupervised | 20.0% | 12.0% | 16.0% | 16.0% | 12.0%–20.0% | 3 |
| recall@100, chunk type purity >= 0.5 · unsupervised | 6.0% | 2.0% | 3.0% | 3.7% | 2.0%–6.0% | 3 |
| recall@100, chunk type any purity · unsupervised | 12.0% | 7.0% | 9.0% | 9.3% | 7.0%–12.0% | 3 |
| recall@200, chunk type purity >= 0.5 · unsupervised | 3.0% | 1.0% | 1.5% | 1.8% | 1.0%–3.0% | 3 |
| recall@200, chunk type any purity · unsupervised | 7.0% | 4.0% | 5.5% | 5.5% | 4.0%–7.0% | 3 |
| validity: modal yield a component · unsupervised · shuffled | 8.7% | 6.8% | 6.8% | 7.4% | 6.8%–8.7% | 3 |
| validity: mean purity · unsupervised · shuffled | 0.17 | 0.11 | 0.11 | 0.13 | 0.11–0.17 | 3 |
| validity: specific reused unit · unsupervised · shuffled | 0.0% | 0.0% | 0.0% | 0.0% | 0.0%–0.0% | 3 |
| validity: chunk types · unsupervised · shuffled | 46 | 44 | 44 | 45 | 44–46 | 3 |
| validity: modal yield a component · unsupervised · slot_shuffled | 13.2% | 17.9% | 13.7% | 14.9% | 13.2%–17.9% | 3 |
| validity: mean purity · unsupervised · slot_shuffled | 0.28 | 0.24 | 0.29 | 0.27 | 0.24–0.29 | 3 |
| validity: specific reused unit · unsupervised · slot_shuffled | 0.0% | 1.8% | 0.0% | 0.6% | 0.0%–1.8% | 3 |
| validity: chunk types · unsupervised · slot_shuffled | 38 | 56 | 51 | 48 | 38–56 | 3 |
| masked top-1 · token 2-gram | 36.8% | 36.4% | 38.4% | 37.2% | 36.4%–38.4% | 3 |
| masked top-5 · token 2-gram | 66.4% | 67.6% | 67.2% | 67.1% | 66.4%–67.6% | 3 |
| open draws real · token 2-gram | 4.4% | 4.5% | 4.9% | 4.6% | 4.4%–4.9% | 3 |
| open draws real unlearned · token 2-gram | 2.4% | 2.3% | 2.2% | 2.3% | 2.2%–2.4% | 3 |
| cue bits, all queries · token 2-gram | 20.15 | 20.47 | 21.37 | 20.67 | 20.15–21.37 | 3 |
| masked top-1 · token 3-gram | 52.6% | 53.6% | 53.0% | 53.1% | 52.6%–53.6% | 3 |
| masked top-5 · token 3-gram | 73.0% | 72.2% | 71.8% | 72.3% | 71.8%–73.0% | 3 |
| open draws real · token 3-gram | 8.6% | 9.5% | 8.3% | 8.8% | 8.3%–9.5% | 3 |
| open draws real unlearned · token 3-gram | 4.2% | 4.5% | 3.5% | 4.0% | 3.5%–4.5% | 3 |
| cue bits, all queries · token 3-gram | 19.53 | 20.18 | 21.27 | 20.32 | 19.53–21.27 | 3 |
| masked top-1 · GRU (best validation size) | 54.4% | 57.2% | 57.6% | 56.4% | 54.4%–57.6% | 3 |
| masked top-5 · GRU (best validation size) | 72.8% | 71.6% | 73.4% | 72.6% | 71.6%–73.4% | 3 |
| open draws real · GRU (best validation size) | 22.8% | 24.8% | 23.9% | 23.8% | 22.8%–24.8% | 3 |
| open draws real unlearned · GRU (best validation size) | 10.9% | 11.5% | 11.2% | 11.2% | 10.9%–11.5% | 3 |
| cue bits, all queries · GRU (best validation size) | 13.73 | 14.13 | 14.94 | 14.27 | 13.73–14.94 | 3 |
| masked top-1 · TRELLIS relational | 18.8% | 24.6% | 24.6% | 22.7% | 18.8%–24.6% | 3 |
| masked top-5 · TRELLIS relational | 43.6% | 48.0% | 48.0% | 46.5% | 43.6%–48.0% | 3 |
| open draws real · TRELLIS relational | 7.3% | 8.3% | 6.6% | 7.4% | 6.6%–8.3% | 3 |
| open draws real unlearned · TRELLIS relational | 6.3% | 6.7% | 5.5% | 6.2% | 5.5%–6.7% | 3 |
| cue bits, all queries · TRELLIS relational | 18.78 | 19.09 | 20.24 | 19.37 | 18.78–20.24 | 3 |
| masked top-1 · TRELLIS supervised | 50.0% | 52.8% | 49.2% | 50.7% | 49.2%–52.8% | 3 |
| masked top-5 · TRELLIS supervised | 68.8% | 70.6% | 69.8% | 69.7% | 68.8%–70.6% | 3 |
| open draws real · TRELLIS supervised | 14.7% | 15.8% | 14.3% | 14.9% | 14.3%–15.8% | 3 |
| open draws real unlearned · TRELLIS supervised | 8.5% | 9.1% | 6.0% | 7.9% | 6.0%–9.1% | 3 |
| cue bits, all queries · TRELLIS supervised | 15.30 | 15.32 | 16.45 | 15.69 | 15.30–16.45 | 3 |
| masked top-1 · TRELLIS unsupervised | 50.8% | 49.2% | 53.0% | 51.0% | 49.2%–53.0% | 3 |
| masked top-5 · TRELLIS unsupervised | 68.8% | 69.6% | 70.4% | 69.6% | 68.8%–70.4% | 3 |
| open draws real · TRELLIS unsupervised | 11.2% | 11.4% | 10.5% | 11.0% | 10.5%–11.4% | 3 |
| open draws real unlearned · TRELLIS unsupervised | 6.4% | 6.1% | 5.6% | 6.0% | 5.6%–6.4% | 3 |
| cue bits, all queries · TRELLIS unsupervised | 15.93 | 16.92 | 17.60 | 16.82 | 15.93–17.60 | 3 |
| cue: exact queries | 48 | 44 | 42 | 45 | 42–48 | 3 |
| cue bits, exact subset · relational | 15.97 | 16.42 | 16.07 | 16.15 | 15.97–16.42 | 3 |
| cue bits, exact subset · supervised | 13.24 | 12.16 | 14.97 | 13.46 | 12.16–14.97 | 3 |
| cue bits, exact subset · unsupervised | 14.07 | 13.31 | 14.94 | 14.11 | 13.31–14.94 | 3 |
| cue bits, exact subset · 2-gram | 17.22 | 16.65 | 18.50 | 17.45 | 16.65–18.50 | 3 |
| cue bits, exact subset · 3-gram | 17.04 | 15.82 | 19.27 | 17.37 | 15.82–19.27 | 3 |
| cue bits, exact subset · GRU (best validation size) | 12.64 | 11.85 | 13.79 | 12.76 | 11.85–13.79 | 3 |
