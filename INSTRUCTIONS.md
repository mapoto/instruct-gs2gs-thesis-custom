

# process data

ns-process-data metashape --data /home/lucky/dataset/metashape_aligned/Dora_grn/clean/ --xml /home/lucky/dataset/metashape_aligned/Dora_grn/cameras.xml --output-dir /home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Dora_grn --ply /home/lucky/dataset/metashape_aligned/Dora_grn/sparse.ply

# init gauss

ns-train splatfacto --data /home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Dora_grn/ --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False --pipeline.model.use_scale_regularization=True nerfstudio-data --downscale-factor 4 --eval-mode all --center-method none --auto-scale-poses False --orientation-method none

# no points gauss
ns-train splatfacto --vis wandb --data /home/lucky/dataset/thesis_nerfstudio_data/coord_corrected/Ephra_grn/ --experiment-name Ephra_grn_no_points nerfstudio-data --load-3D-points False --downscale-factor 4 --eval-mode all --center-method none --auto-scale-poses False --orientation-method none