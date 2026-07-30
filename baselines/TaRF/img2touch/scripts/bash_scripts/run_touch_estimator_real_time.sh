export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"
python scripts/touch_estimator_real_time.py \
    --indir ../nerfstudio_modules/outputs/touch_estimation_input_cache \
    --bg_path touch_bg/bench_colmap_40_50/bg.jpg\
    --outdir outputs/touch_estimator_real_time/ \
    --ddim_steps 200 \
    --config configs/tarf.yaml \
    --diffusion_ckpt pretrained_models/img2touch.ckpt \
    --ranking_rgb_enc_ckpt pretrained_models/reranking_rgb_enc.ckpt \
    --ranking_tac_enc_ckpt pretrained_models/reranking_tac_enc.ckpt \
    --n_samples 8 \
    --scale 7.5
