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

help : Makefile
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
DEVICE := cpu
EXP_NAME := flower_16bit

## fit: Implicit MLP image fitting
fit:
	${PYTHON} main.py \
 	exp_name=$(EXP_NAME) \
 	device=$(DEVICE) $(KWARGS) $(HYDRA_FLAGS)

all: fit