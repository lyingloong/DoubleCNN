.PHONY: clean clean_all

clean:
	rm -rf ./checkpoints
	rm -rf ./results
	rm -rf ./test_results
	rm ./output.txt

clean_all:
	rm -rf ./checkpoints
	rm -rf ./results
	rm -rf ./test_results
	rm ./result_long_term_forecast.txt
	rm ./output.txt