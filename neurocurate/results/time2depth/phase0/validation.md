# Validation

- Frozen protocol SHA256: `d6c28de39f31319f3c5cf9fdfbb340583f3158ac8949876cb2c701d0d7421ea3`
- Five real EEG inputs verified as float64 `(200,80,17,100)` and loaded with `allow_pickle=False`.
- 200 unique ordered stimulus IDs; local stimulus files present.
- DINOv2 pinned local revision; final-layer 5-image smoke comparison is recorded in `model_layer_schema.json`.
- RDM integrity: `{"brain_shape": [5, 6, 200, 200], "model_shape": [12, 200, 200], "rsa_shape": [5, 6, 12], "brain_all_finite": true, "model_all_finite": true, "brain_symmetric": true, "model_symmetric": true, "all_diagonals_zero": true, "ordered_ids_unique": true}`
- No EEG/checkpoint downloads, training, or previous-result modifications.
