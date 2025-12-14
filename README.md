1. Place your own `phone_ctm.txt` file in project root dir, or use the default one generated
   from https://github.com/tjysdsg/aidatatang_force_align on AISHELL-3 data
2. Run

```bash
python feature_extraction.py
```

to collect required statistics (phone start time, duration, tones, etc). Results are saved to `utt2tones.json`.

Note: this repo includes a precomputed `utt2tones.json`. If you don’t have the real `phone_ctm.txt` (the repo’s copy may be a Git LFS pointer), you can skip step 2.

3. Run

```bash
python train/split_wavs.py
```

to split train, validation, and test dataset for embedding model training

If you have AISHELL-3 wavs locally, set:

```bash
export TONE_WAV_DIR=/path/to/AISHELL-3/SPEECHDATA
```

The test utterances used in the paper are listed in [test_utts.json](test_utts.json)

4. Run

```bash
python train/train_embedding.py --save_dir embedding_exp
```

to train embedding model, the results are in `exp/`

Mel-spectrogram cache is generated at `exp/cache/spectro/wav.scp` and `exp/cache/spectro/*.npy`

## [Optional] Train an end-to-end tone recognizer

After step 3,

- Run the following at `e2e_tone_recog/`

```bash
./run.sh
```
