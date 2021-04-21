# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
HYDRA_FLAGS := -m
USE_WANDB := True
SEED := 0

.PHONY: help docs
.DEFAULT: help

## install: install all dependencies
install:
	pip install -r requirements.txt

setup_zsh:
	@sudo apt-get install -y zsh
	@echo "Yes" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
	@zsh

colab_install: setup_zsh install

help : Makefile makefiles/*.mk
    ifeq ($(UNAME_S),Linux)
		@sed -ns -e '$$a\\' -e 's/^##//p' $^
    endif
    ifeq ($(UNAME_S),Darwin)
        ifneq (, $(shell which gsed))
			@gsed -sn -e 's/^##//p' -e '$$a\\' $^
        else
			@sed -n 's/^##//p' $^
        endif
    endif

# Config defaults
EXP_NAME := flower_16bit

## fit: Implicit MLP image fitting
siren:
	${PYTHON} implicit_image/compress.py \
 	exp_name=$(EXP_NAME) \
 	$(KWARGS) $(HYDRA_FLAGS)

MASKING := RigL
DENSITY := 0.5
TRAIN_MUL := 5
prune:
	${PYTHON} implicit_image/compress.py \
	+masking=$(MASKING) masking.density=$(DENSITY) train.multiplier=$(TRAIN_MUL) \
 	exp_name='$${masking.name}_$${masking.density}_trainx_$${train.multiplier}' \
 	$(KWARGS) $(HYDRA_FLAGS)

include makefiles/*.mk

all: siren prune