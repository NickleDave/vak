CONFIG_FILE=./tests/setup_scripts/Makefile_config.ini
TMP_CONFIG_FILE=./tests/setup_scripts/tmp_Makefile_config.ini
CBIN_DIR=./tests/test_data/cbins/gy6or6/032312
CBINS=$(wildcard $(CBIN_DIR)/*.cbin)
NOTDIR_CBINS=$(notdir $(CBINS))
SPECT_FILES=$(patsubst %.cbin, ./tests/test_data/spects/%.cbin.spect, $(NOTDIR_CBINS))
SPECTS_SCRIPT=./tests/setup_scripts/remake_spects.py
RESULTS_SCRIPT=./tests/setup_scripts/remake_results.py

.PHONY: variables clean download config data results all

variables:
	@echo CBIN_DIR: $(CBIN_DIR)
	@echo CBINS: $(CBINS)
	@echo SPECT_FILES: $(SPECT_FILES)
	@echo SPECTS_SCRIPT: $(SPECTS_SCRIPT)
	@echo RESULTS_SCRIPT: $(RESULTS_SCRIPT)

clean :
	rm -rf ./tests/test_data/results/*
	rm -rf ./tests/test_data/spects/*
	rm $(TMP_CONFIG_FILE)

download :
	wget -O ./tests/test_data/spects/spects.tar.gz "https://ndownloader.figshare.com/files/14554052"
	tar -xvf ./tests/test_data/spects/spects.tar.gz -C ./tests/test_data/spects/
	wget -O ./tests/test_data/results/results.tar.gz "https://ndownloader.figshare.com/files/14554049"
	tar -xvf ./tests/test_data/results/results.tar.gz -C ./tests/test_data/results/

$(TMP_CONFIG_FILE) : $(CONFIG_FILE)
	cp  $(CONFIG_FILE) $(TMP_CONFIG_FILE)

config : $(TMP_CONFIG_FILE)

$(SPECT_FILES) : config $(CBINS) $(SPECTS_SCRIPT)
	python $(SPECTS_SCRIPT)

data : $(SPECT_FILES)

results : data $(RESULTS_SCRIPT)
	python $(RESULTS_SCRIPT)

all : results


