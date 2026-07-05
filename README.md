<!--
  README — ganti USERNAME/REPO di seluruh badge & link Colab dengan
  username GitHub dan nama repository Anda yang sebenarnya.
-->

# Digital Rock Physics: Micro-CT Restoration & Pore Segmentation

Perbandingan tiga skema *pipeline* untuk restorasi citra micro-CT batuan dan
segmentasi pori menggunakan *backbone* **Residual U-Net 2D/2.5D**. Repositori
ini berisi seluruh kode (notebook Colab + modul Python) untuk Tugas Akhir di
bidang **Digital Rock Physics**.

Tujuan penelitian: menentukan *trade-off* terbaik antara **akurasi**,
**efisiensi komputasi**, dan **modularitas** dalam merestorasi serta
mensegmentasi citra micro-CT batuan.

---

## Ringkasan Penelitian

Penelitian disusun dalam dua fase kumulatif:

- **TA 1 — Studi Pendahuluan.** Membandingkan 2D U-Net, 3D U-Net, dan SCU-Net
  untuk *denoising* pada sampel batupasir *Mount Simon* dengan derau
  *salt-and-pepper*.
- **TA 2 — Studi Utama.** Membandingkan tiga skema *pipeline* restorasi +
  segmentasi, dievaluasi pada sampel **Libo**, **Bentheimer**, dan **PTP**.

Ketiga skema berbeda dalam cara tahap restorasi dan segmentasi digabungkan:

| Skema | Nama | Arsitektur | Karakteristik |
|:-----:|------|-----------|---------------|
| **K1** | Cascade modular penuh | Blok R → Blok NS → Blok B | Kualitas restorasi terbaik; paling modular |
| **K2** | Gabungan sebagian | Blok RNS gabungan + Blok B terpisah | Restorasi setara K1, latih ~2.5× lebih cepat |
| **K3** | Monolitik *end-to-end* | Blok NRSB tunggal | Gagal pada domain karbonat asing (PTP) |

Keterangan blok: **R** = *de-ringing*, **NS** = *denoising* + koreksi *rotation-step*,
**B** = segmentasi biner (pori/padatan).

---

## Hasil Utama

**TA 1 — Denoising (Mount Simon)**

| Model | Porositas | Galat relatif | Waktu denoising |
|-------|:---------:|:-------------:|:---------------:|
| **2D U-Net** | 18.06 % | 3.02 % | ~2.75 s |
| 3D U-Net | — | — | ~19.83 s |

2D U-Net memberikan akurasi porositas kompetitif dengan waktu inferensi ~7×
lebih cepat dibanding 3D U-Net.

**TA 2 — Perbandingan Skema**

- **K1** unggul pada seluruh metrik restorasi (keunggulan PSNR **0.12–0.45 dB**
  atas K2).
- **K2** melatih **~2.5× lebih cepat** dengan kualitas restorasi setara.
- **K3** gagal pada sampel karbonat *out-of-distribution* (PTP): artefak *ring*
  bocor ke prediksi, *over-smoothing* menurunkan SSA sehingga permeabilitas
  Kozeny-Carman menjadi *inflated* (galat porositas *over-segmentation* ~+33.2 %).

> **Catatan.** Kegagalan K3 pada PTP adalah bukti empiris yang diharapkan: ia
> menunjukkan mengapa pemisahan tahap *de-ringing* diperlukan. Artefak *ring*
> berasal dari detektor dan bersifat global, sehingga *receptive field* terbatas
> pada K3 tidak dapat menangkapnya.

Metrik yang dilaporkan — restorasi: PSNR, SSIM, RMSE · segmentasi: Dice, IoU,
akurasi · petrofisika: porositas, SSA, permeabilitas Kozeny-Carman.

---

## Struktur Repositori

```
.
├── notebooks/
│   ├── TA1_preliminary/      # Studi pendahuluan (denoising)
│   │   ├── 2D_U_Net_Complete.ipynb
│   │   └── 3D_U_Net_Complete.ipynb
│   ├── K1_cascade/           # Skema K1 — cascade modular
│   │   ├── K1_blokR_ResUNet2D.ipynb            # training Blok R (de-ring)
│   │   ├── K1_blokNS_ResUNet2D.ipynb           # training Blok NS (denoise)
│   │   └── K1_cascade_inference_ResUNet2D.ipynb
│   ├── K2_partial/           # Skema K2 — RNS gabungan
│   │   └── K2_endtoend_inference_ResUNet2D.ipynb
│   ├── K3_monolithic/        # Skema K3 — NRSB end-to-end
│   │   └── ResidualUNet_PoreSegmentation_v2.ipynb
│   └── segmentation/         # Blok B — segmentasi pori 2.5D
│       ├── resunet_segmentation_colab_2p5D.ipynb
│       └── resunet_segmentation_inference_colab.ipynb
├── requirements.txt
```

## Arsitektur
- **Residual U-Net 2D** (8.28 M parameter) — dasar untuk Blok R (*de-ringing*) dan Blok NS (*denoising*).
- **Residual U-Net 2.5D** (input 5-kanal, keluaran 2-kelas *softmax*) — Blok B
  dan skema K3.

Referensi utama: **Ronneberger et al. 2015** (arsitektur U-Net) dan
**Wang et al. 2024** (*SPE Journal* — U-Net 2.5D untuk segmentasi digital rock);
lihat [Referensi](#referensi). Detail lengkap arsitektur:
[`docs/architecture.md`](docs/architecture.md).

---

## Referensi

Referensi utama yang mendasari arsitektur dan metodologi repositori ini:

1. **Wang, H., Guo, R., Dalton, L. E., Crandall, D., Hosseini, S. A., Fan, M.,
   & Chen, C. (2024).** *Comparative Assessment of U-Net-Based Deep Learning
   Models for Segmenting Microfractures and Pore Spaces in Digital Rocks.*
   SPE Journal, SPE-215117-PA.
   [https://doi.org/10.2118/215117-PA](https://doi.org/10.2118/215117-PA)

2. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** *U-Net: Convolutional
   Networks for Biomedical Image Segmentation.* MICCAI 2015.
   [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) ·
   [https://doi.org/10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)

<details>
<summary>Format BibTeX</summary>

```bibtex
@article{wang2024comparative,
  title   = {Comparative Assessment of U-Net-Based Deep Learning Models for
             Segmenting Microfractures and Pore Spaces in Digital Rocks},
  author  = {Wang, Hongsheng and Guo, Ruichang and Dalton, Laura E. and
             Crandall, Dustin and Hosseini, Seyyed A. and Fan, Ming and
             Chen, Cheng},
  journal = {SPE Journal},
  year    = {2024},
  note    = {SPE-215117-PA},
  doi     = {10.2118/215117-PA}
}

@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention
               (MICCAI)},
  pages     = {234--241},
  year      = {2015},
  publisher = {Springer},
  doi       = {10.1007/978-3-319-24574-4_28},
  eprint    = {1505.04597},
  archivePrefix = {arXiv}
}
```
