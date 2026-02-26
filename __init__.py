"""
ml_analysis: Publication-quality ML pipeline for pose-based ASD prediction.

Builds on pose_analysis correlation results with:
  - Rich feature extraction from frame-level data (11 statistics per metric)
  - Dimensionality reduction (PCA + UMAP) for visual exploration
  - Nested cross-validated classification (ASD vs TD)
  - Nested cross-validated regression (ADOS-2 total score)
  - SHAP explainability for best models

Usage:
    python -m ml_analysis.run_ml \\
        --csv      /path/to/child_for_humanlisbet_paper_with_paths.csv \\
        --pose-records /path/to/pose_records/ \\
        --output-dir ml_analysis/results/
"""
