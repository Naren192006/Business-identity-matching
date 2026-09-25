@echo off
cd /d "%~dp0"
echo === test S1 start %date% %time% === >> prep_test.log
python3 -X utf8 src/preprocess.py test S1 dataset/test/test_source1.tsv >> prep_test.log 2>&1
echo === test S2 start %date% %time% === >> prep_test.log
python3 -X utf8 src/preprocess.py test S2 dataset/test/test_source2.tsv >> prep_test.log 2>&1
echo === test S3 start %date% %time% === >> prep_test.log
python3 -X utf8 src/preprocess.py test S3 dataset/test/test_source3.tsv >> prep_test.log 2>&1
echo === test assemble start %date% %time% === >> prep_test.log
python3 -X utf8 src/preprocess.py test assemble >> prep_test.log 2>&1
echo === test prep DONE %date% %time% === >> prep_test.log
