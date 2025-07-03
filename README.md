# AI4Health Summer School 2025 - Practical Session on Diffusion Models

| **[Abstract](#abstract)**
| **[Links](#links)**
| **[Table of Content](#table-of-content)**
| **[Diagram](#diagram)**

This is the Git repository for the practical session on diffusion models for the 2025 edition of the AI4Health Summer School.

**Authors**: Hugues Roy, Maelys Solal, Ninon Burgos

Many thanks to Manon Heffernan, Charlotte Godard and Swann Ruyter for their invaluable feedback.

## Abstract

This practical session introduces diffusion models and an application in medical imaging. Participants will first gain a conceptual and mathematical understanding of diffusion models through simple, illustrative examples. From there, they will implement a diffusion model from scratch using PyTorch. Key topics covered include time embeddings, conditioning the U-Net architecture, implementing variance schedules, and modeling both the forward and reverse diffusion processes. The session also examines alternative modeling strategies such as data prediction, noise prediction, and score-based approaches. Participants will train and evaluate their models on medical imaging datasets, providing practical insights into their effectiveness in clinical contexts. Finally, students will work on an application for anomaly detection, specifically synthesising pseudo-healthy brain MRIs to detect tumors.

## Links

The practical session is based on a [Google Colab notebook](https://colab.research.google.com/drive/15SMtpuetkRqI07XZj3h2_ZE5-9aT4gM4?usp=sharing), available as read-only. Please copy this notebook in your own Google Drive so that you can edit it.

If you have any questions at any point during the practical session, please either raise your hand or type in your questions on the [Speakup](https://web.speakup.info/room/join/71824). You can also upvote and comment on questions from other participants.

During the last part of the lab session (section 5. Application to anomaly detection), you are invited to log the results of your models on the following [Google Sheet](https://docs.google.com/spreadsheets/d/1mXgYuupxjtmwF0A-fIIntp0wJ_F_LoIzj91Hrv3C40A/edit?usp=sharing).

At the end of the practical session, please provide us feedback using the following [Google Form](https://forms.gle/m8oVfcfCFew2qGqr6).

## Table of Content

The lab session is divided into the following sections:

1. Mathematical formulation of diffusion models
2. Dataset manipulation
3. Crash course on PyTorch (optional)
4. Implementation of all components necessary for training a diffusion model:
    (a) U-Net and time embeddings,
    (b) Beta schedule,
    (c) Forward process,
    (d) Reverse process (from different perspectives)
5. Application to anomaly detection with the anoDDPM model

## Diagram

<img src='https://drive.google.com/uc?id=11mjeMdwiTiPiAYgXsagEKAVW2iTILs7b'>
