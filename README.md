PS12 – Underwater Domain Awareness (Stride-Based Pipeline)
This repository contains the inference pipeline used for Stage-1 Event Detection and Stage-2 Event Classification for the AI Grand Challenge (Problem Statement 12 – Underwater Domain Awareness).
The pipeline implements a 5-second sliding window with a 2-second stride, performs signal preprocessing, two-stage ML classification, smoothing, merging, 
and generates the final annotations JSON in the required Grand Challenge format.

ps12_stride_pipeline/
│
├── pipeline_stride.py        # Main inference script
├── stage1_xgb_model.pkl      # Background/Event classifier
├── stage1_scaler.pkl         # Scaler used for Stage-1 model
├── stage2_xgb_model.pkl      # 4-class event classifier
├── stage2_scaler.pkl         # Scaler used for Stage-2 model
├── stage2_label_encoder.pkl  # Label encoder for class mapping
├── sample_audios/            # (Optional) Few sample files for testing
└── README.md                 # Documentation
