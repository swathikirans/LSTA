python main_rgb.py --dataset gtea_61 --root_dir dataset --outDir experiments --stage 1 \
                   --seqLen 25 --trainBatchSize 32 --numEpochs 200 --lr 0.001 --stepSize 25 75 150 \
                   --decayRate 0.1 --memSize 512 --outPoolSize 100 --evalInterval 5 --split 2