## vis.jpeg_plots: plot JPEG PSNR, size vs quality
vis.jpeg_plots:
	${PYTHON} vis_tools/jpeg_quality_curve.py

## vis.weight_removal: plot results of weight reduction techniques
vis.weight_removal:
	${PYTHON} vis_tools/weight_removal.py wandb.project=sparsify

VIS_DEPS := vis.jpeg_plots
VIS_DEPS += vis.sparsify

## vis: All visualizations
vis: $(VIS_DEPS)