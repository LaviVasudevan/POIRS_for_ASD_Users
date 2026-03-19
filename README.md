# Point of Interest Recommender for Autistic Users Using Sensory-Aware Graph Attention and RNN 

> A multimodal, heterogeneous knowledge graph-based recommendation system designed to provide safe, comfortable, and sensory-aware public space navigation for individuals on the Autism Spectrum.

---

## Table of Contents
1. [Project Overview](#-project-overview)
4. [Repository Structure](#-repository-structure)
5. [Dataset Sources](#-dataset-sources)
6. [System Requirements](#-system-requirements)
7. [Setup Instruction](#-setup-instruction)

---

## Project Overview
Navigating public spaces can often be overwhelming for individuals on the autism spectrum due to unpredictable sensory environments. This project introduces a Point of Interest Recommender System (POIRS) that models venues not just by location or popularity, but by their **sensory attributes** (noise, lighting, crowding, etc.). By utilizing Graph Attention Networks (GAT) and RNNs, the system matches user sensory profiles with safe, accessible environments.

---

## Repository Structure

```text
POIRS_for_ASD_Users/
├── backend/
│   ├── data/                 # Local data storage 
│   ├── templates/            # HTML/UI templates
│   ├── app.py                # Main Flask/FastAPI backend application
│   ├── Dockerfile            # Containerization setup
│   └── requirements.txt      # Backend Python dependencies
├── pipeline/
│   ├── CKG_Construction...   # Scripts for Knowledge Graph building
│   ├── CKG_Visualization...  # Graph visualization tools
│   └── retrain_pipeline.py   # Main GAT training and embedding script
├── data/
│   ├── images.jsonl          # Processed image metadata
│   ├── pois.jsonl            # POI metadata and categories
│   └── reviews.jsonl         # Textual review data
└── README.md
```

## Dataset Sources

The dataset utilized for this project is a multimodal collection stored primarily via MongoDB and locally within the `data/` directory. It includes:

* **POI Metadata:** 583 venues across 8 categories, sourced from Google Places API and TripAdvisor API.
* **User Sensory Profiles:** 210 individual responses.
* **Textual Reviews:** 51,649 records processed using VADER for sentiment polarity.
* **Image Metadata:** 1,396 images processed using LLaVA for fine-grained sensory attribute extraction and captioning.

---

## System Requirements

### Hardware Requirements

**Cloud Computing (Kaggle) - For Model Training:**
* **GPU:** Dual NVIDIA Tesla T4 (15 GiB per GPU, 30 GiB total)
* **Architecture:** Turing (CUDA Compute Capability: 7.5)
* **Purpose:** Hardware acceleration for heterogeneous GAT training, TransR embedding, and Automatic Mixed Precision (AMP) training.

**Local Development Environment:**
* **OS:** Windows 11 (Build 26200.7019)
* **Processor:** 12th Gen Intel Core i7-1260P (12 cores, 16 threads)
* **Memory:** 16 GB RAM
* **IDE:** Visual Studio Code

### Software & Framework Requirements

* **Python Version:** 3.11
* **Database:** MongoDB (for flexible schema management of multimodal data)
* **Deep Learning Ecosystem:**
  * PyTorch 2.6.0+cu124
  * PyTorch Geometric 2.7.0 (HeteroData, LinkNeighborLoader)
  * PyTorch Scatter 2.1.2, Sparse 0.6.18, Cluster 1.6.3, Spline Conv 1.2.2, PyG-Lib 0.5.0
* **Scientific Computing:**
  * NumPy 1.26.4, SciPy 1.15.3, Pandas
  * Intel MKL 2025.3.0, Intel OpenMP 2024.2.0, Threading Building Blocks (TBB) 2022.3.0

---

## Setup Instruction

**Step 1: Clone the Repository**
```bash
git clone https://github.com/LaviVasudevan/POIRS_for_ASD_Users.git
cd POIRS_for_ASD_Users
```

**Step 2: Install Dependencies**

Ensure your environment matches the required PyTorch + CUDA versions, then install the required packages:
```bash
pip install -r backend/requirements.txt
```

**Step 3: Execute the Training Pipeline**

Run the pipeline to construct the Knowledge Graph, execute GAT encoding, and optimize embeddings:
```bash
cd pipeline
python retrain_pipeline.py
```

**Step 4: Start the Backend Server**

Launch the server to serve the sensory-aware recommendations to the frontend interface:
```bash
cd ../backend
python app.py
```
