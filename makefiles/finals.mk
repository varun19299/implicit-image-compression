# final.compress.%: Compress image. % in flower, building, bridge
EXP_NAME = 'density_$${masking.density}'
finals.compress.%:
	${PYTHON} implicit_image/compress.py \
	img=$* \
	masking.density=$(DENSITY) \
	entropy_coding=zstd \
	train.multiplier=5 \
	exp_name=$(EXP_NAME) \
	wandb.name='$${img.name}_$${exp_name}' wandb.project=finals_simple \
	$(KWARGS) $(HYDRA_FLAGS)