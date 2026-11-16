# Sources of the copied training scripts

Every file in `training/` is a copy of a script or record **as run, not re-run here**. "Original" is the path relative to
our working folder at the time (`../` = a sibling folder of it). "SHA-256" is the hash of the original bytes, before the
privacy clean-up. "Changed" says what the clean-up changed; "no" means the copy is byte-identical to the original.

## How the originals were matched to the runs

- **U-Nets:** `train_cuda.py`, `prepare_train.py`, `run_train.py` and `views.py` have the hashes recorded for the fold 0, 3 and 4
  training notebooks; the fold 1 and fold 2 runs record the same `train_cuda.py` hash. `wave.json` and `protocol.json`
  are the files embedded in the fold 0 notebook (hashes equal to its record). `fold5mean.py`, `test_maps.py` and
  `audit_fold_models.py` made the averaged test maps.
- **Mask2Former:** `train.py`, `model.py` and `geom.py` are our working copies, edited after the final runs to add opt-in
  switches that are off by default (tests check that the default config, loss and draws are unchanged); the bytes at
  run time were not hashed. The exporters, evaluator and scorer predate the runs that used them. `model_config.json`
  and `train_done_c1536_f0.json` are the records of the fold 0 run.
- **Filter and repair:** the final fitting run recorded the hash of each of its 154 input files. The 96 that are not
  package files are copied here (two evidence files over 1 MB are listed but not copied), each with the recorded hash.
  The other 58 are this repository's `src/filseg` at an earlier development commit (`3f6524cd`, not part of this repository's public history) and are not copied.
  `finish.py`, `stage.py`, `render_kernel.py`, `paired_vs_ref.py` and `cli_ms125.json` predate the run that used them.

## Counts

| Folder | Files listed | Copied | Copies that differ from the original |
| --- | ---: | ---: | ---: |
| `mask2former/` | 24 | 24 | 4 |
| `unet/` | 9 | 9 | 2 |
| `selector_and_repair/` | 102 | 100 | 18 |

## Files

| Copied path | Original | SHA-256 of the original | Status | Changed |
| --- | --- | --- | --- | --- |
| `mask2former/train.py` | `a1native/train.py` | `4a11b21b4405703e5608e8a4a091450fd9628eba5628dbd42129297b40d40a7c` | as run, not re-run here | df_path default: env MF_DF_PATH or "." (was the host home). A new run records the new value in its cfg JSON, so its cfg hash differs from the as-run runs; tests/test_gc1_flags.py pins the old cfg hashes and fails on them. |
| `mask2former/model.py` | `a1native/model.py` | `2a1c7206a0c472c50abe7ddc5ebb2c5832d611d4debfa9b4f560b1407aaea5f7` | as run, not re-run here | no |
| `mask2former/geom.py` | `a1native/geom.py` | `afb161173c49c966eca0a5fbe3dd65ba838dfb9a3825c6fc823d14ccbfc3fa32` | as run, not re-run here | no |
| `mask2former/decode.py` | `a1native/decode.py` | `a67f7987beda78fd5c0fa51fb74aad73c94000a4f11e399cf809e18882a0d1f7` | as run, not re-run here | no |
| `mask2former/eval_ckpt.py` | `a1native/eval_ckpt.py` | `b01ac37a513fcf35b714ad8a6151e52f8a0a07c017e35a04bd989848d4f20e54` | as run, not re-run here | no |
| `mask2former/export_tta.py` | `a1native/export_tta.py` | `7ebfe9e92f0e6b3e841e9d72fc874a19a580be432a09847b60fd330892910bdc` | as run, not re-run here | no |
| `mask2former/pqscore.py` | `a1native/pqscore.py` | `4cadb984f02257973388fdf6a5690cfff7445b7f82cdbf710d318159106e8ade` | as run, not re-run here | no |
| `mask2former/score.py` | `a1native/score.py` | `9c1ba0be7561c55090baa8c17fa2d38e096dafe451aec521a12fd8b6c868502d` | as run, not re-run here | no |
| `mask2former/fold_table_time24h_k5_seed42.json` | `a1native/fold_table_time24h_k5_seed42.json` | `2a2451b3d4d9420137ac3e9c6ecd97a7741100ecf35f8accdc0d2140138118f8` | as run, not re-run here | no |
| `mask2former/export_scale.py` | `ms1/export_scale.py` | `ae80a3beee0956321f19fe6f5b00360dfc5e3f37b5881383304575418c8875bd` | as run, not re-run here | no |
| `mask2former/model_config.json` | `a1native/artifacts/c1536_f0/model_config.json` | `2d1fdfb195b81f0f335bca028c20643ebc44e33efdd612d8d7fa2d112d5f3a60` | as run, not re-run here | no |
| `mask2former/train_done_c1536_f0.json` | `a1native/artifacts/c1536_f0/train_done.json` | `fa5d552136c6d60379fe5df4f6f916dbd5fdfb59d5830cdf40fb7c1cc029768d` | as run, not re-run here | record only: host path replaced |
| `mask2former/tests/conftest.py` | `a1native/tests/conftest.py` | `8659c4f6af34a32f61ecc29774fd873cf628e7e0828ce9ff04773418357a6cc6` | as run, not re-run here | no |
| `mask2former/tests/test_consensus.py` | `a1native/tests/test_consensus.py` | `6f5c7bcd700bf22ad3339ede55bad751fdb57685366413ee75b872d7957c6229` | as run, not re-run here | no |
| `mask2former/tests/test_decode.py` | `a1native/tests/test_decode.py` | `dc1b4a972185eed8f3513f5af532c5c7384cad46950c30259c6ff6f36d4008ec` | as run, not re-run here | no |
| `mask2former/tests/test_export_tta.py` | `a1native/tests/test_export_tta.py` | `8e00606fd5bf4ae314a2dcff23621539c063c6be817834e0fcca4acb881ba680` | as run, not re-run here | path to the package module made relative to this repository |
| `mask2former/tests/test_fold_guards.py` | `a1native/tests/test_fold_guards.py` | `03beb19e205411b93496553d1e9d3597bc8af6f615ff0ab1441a61c289329542` | as run, not re-run here | no |
| `mask2former/tests/test_gc1_ckpt_hf.py` | `a1native/tests/test_gc1_ckpt_hf.py` | `d6734731e0bbb53c021fb9df63ef7d36a7e83e71e4c9a794d4bb6b8e058c15ff` | as run, not re-run here | no |
| `mask2former/tests/test_gc1_finetune.py` | `a1native/tests/test_gc1_finetune.py` | `d2dfd6c5a1c456c069fcc413579e16d81bcaaff0fe24d434a9acbdc7170479d4` | as run, not re-run here | no |
| `mask2former/tests/test_gc1_flags.py` | `a1native/tests/test_gc1_flags.py` | `593ebd4ca635bdd60711ffd690fec293bba77ac44c5fa6518db384358663602f` | as run, not re-run here | no |
| `mask2former/tests/test_geom.py` | `a1native/tests/test_geom.py` | `036ccff919e69c4061a380ba31ed6c96c845208bc11cef736a2192342b1d8386` | as run, not re-run here | no |
| `mask2former/tests/test_pqscore.py` | `a1native/tests/test_pqscore.py` | `c7d1a6a194352748e5ef0c19c45484be7ccdf86b7ddcc75463ad2f09518b758c` | as run, not re-run here | no |
| `mask2former/tests/test_screening.py` | `a1native/tests/test_screening.py` | `0539729e6ce741e07c5f97dad670df61dce80d80e8f9ca9781d44d7d4f0ffa2a` | as run, not re-run here | path to the package made relative to this repository |
| `mask2former/tests/test_train.py` | `a1native/tests/test_train.py` | `73d232036d663a8753b8e71e2069d3c377a853803b81b1db45c2135173211a4c` | as run, not re-run here | no |
| `unet/train_cuda.py` | `takeover/sync/detect_20261110/delivery_prep/staging/campaigns/20261110_delivery_prep/source_snapshot/detect_20261106/oof_bank5/frozen/train_cuda.py` | `c241c76c1aabcdcaf836e8bfd976f8bd9b3c565d13ebd95e8772472af7416de9` | as run, not re-run here | no |
| `unet/views.py` | `takeover/sync/detect_20261110/delivery_prep/staging/campaigns/20261110_delivery_prep/source_snapshot/detect_20261106/oof_bank5/frozen/views.py` | `ed37e0c4cb31b1b0017facd012b2125d5d2dbf5413bf981c835bff0614ca96b8` | as run, not re-run here | no |
| `unet/prepare_train.py` | `../kaggle-playground-harness-migrate/filament/relabel_pilot_20261031/prepare_train.py` | `8d8ed2e638010a884350761d64b4029c9af111154116ffd856dacd03a7f6d0db` | as run, not re-run here | no |
| `unet/run_train.py` | `../kaggle-playground-harness-migrate/filament/relabel_pilot_20261031/run_train.py` | `8ced407cac2abc3aa86ef2e83f9a832eba6318770c3b45baff3e24b176da0809` | as run, not re-run here | no |
| `unet/fold5mean.py` | `unet_refit/fold5mean.py` | `d2eb5f091e83b0fd96dbe57064786c6d779ce50f1f037cc6c2280ba6afbe05ca` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `unet/test_maps.py` | `unet_refit/test_maps.py` | `5ed75f79df32edbd8e76a45abca09bfa6af9fa17b30bc049fb01aecfea2a23da` | as run, not re-run here | no |
| `unet/audit_fold_models.py` | `unet_refit/audit_fold_models.py` | `9f01ce19bd860cee0c187a5b70b6b884b4a6ec169679cdb4ec9ee8b3e989dc86` | as run, not re-run here | input roots from env UNET_RUNS_ROOT / TEST_IMAGES or this repository (were private absolute paths); adds "import os" |
| `unet/wave.json` | `../kaggle-playground-harness-migrate/filament/detect_20261103/p0_fold0/staged/filament-p0-fold0-gpu-1103/run.ipynb (embedded file wave.json)` | `355e574f8affec27f0ee4690dff56d0836ce37db9c217c738c3232c47690e56d` | as run, not re-run here | no |
| `unet/protocol.json` | `../kaggle-playground-harness-migrate/filament/detect_20261103/p0_fold0/staged/filament-p0-fold0-gpu-1103/run.ipynb (embedded file protocol.json)` | `22fd40b57dfeb3ef81c6dc7e3a49a89b02720ec62896a58e91ffa8cb167dcb56` | as run, not re-run here | no |
| `selector_and_repair/case/bank_loader.py` | `chainF/case/bank_loader.py` | `f5ac4d60768651ff0affae71e25f70d5595b7db993e80f49c3caf2180100a495` | as run, not re-run here | no |
| `selector_and_repair/case/chain_inputs.json` | `chainF/host/stage/ms125/chain_inputs.json` | `9ba80fc254c74be27021c397df700c5c3d90d519361a68fd091ad73b78870185` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/context_features.py` | `chainF/case/context_features.py` | `313e87b790d62e00316dbaff73f8521bc07c0af6c8bb0349ea3f74eb2f0a191e` | as run, not re-run here | no |
| `selector_and_repair/case/count_plan.py` | `chainF/case/count_plan.py` | `edde4b1618cd79a0fc3728b65d544df8b43817ca46b8c44ad9f78927ba8f69e4` | as run, not re-run here | no |
| `selector_and_repair/case/evidence/archm2f_fold2_MF_protocol.json` | `chainF/case/evidence/archm2f_fold2_MF_protocol.json` | `98baa4e4e8e25c7a0253544247e3c7aa791739695f4b3e4f62dc5c617180384c` | as run, not re-run here | no |
| `selector_and_repair/case/evidence/fold2_map_provenance.json` | `chainF/case/evidence/fold2_map_provenance.json` | `9ce5b871112d4f17d3f18fc83bd6eb4f583ce0354c64ccc00d162dbee88cb636` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/evidence/mfhires_stage0_archives.json` | `chainF/case/evidence/mfhires_stage0_archives.json` | `f7bb4103bd2d7cb3c478e53b17ae860cd494894ea15febb15ad5868ffaaf31c2` | as run, not re-run here | no |
| `selector_and_repair/case/evidence/mftta_stage_a.json` | `chainF/case/evidence/mftta_stage_a.json` | `2058c25ca0857cfd94e52a7cfda6ea0e4408635b99de329ee7a6704236cea8fc` | as run, not re-run here | no |
| `selector_and_repair/case/evidence/original_full_raw_1024_f2_s42.json` | `chainF/case/evidence/original_full_raw_1024_f2_s42.json` | `e59ae5d9c99d7a5ed486890ea656541c3951b6efb956f185a608a4df7667a0ed` | as run, not re-run here | no |
| `selector_and_repair/case/evidence/original_local_raw_clahe_detail_f2_s42.json` | `chainF/case/evidence/original_local_raw_clahe_detail_f2_s42.json` | `eb783f6b3e11a7c3978097cd6f76f6b70fffee4ec97da3c6ccfda41081ba24b8` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/contract.py` | `chainF/case/frozen/contract.py` | `bcc86318ca1bd81cf0a565353bb2c6d648e794b4ed16408f7e31055bf6197b11` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/__init__.py` | `chainF/case/frozen/eval/filseg/__init__.py` | `150dd4dc812e96a70b6410fb20adc73a1f64bffe22b56d07192f5e5e15472d27` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/acceptance.py` | `chainF/case/frozen/eval/filseg/acceptance.py` | `3c719308517eab5ee5b55fb39613e2965aaadae03da77ac5ef7352a21f3b9b7e` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/cli.py` | `chainF/case/frozen/eval/filseg/cli.py` | `2502b1516370a54d227f33d9c6c8bab1e76132142a59abdae77798dbb32023b7` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/coco_io.py` | `chainF/case/frozen/eval/filseg/coco_io.py` | `69d0aaa351c6753b412d111251b0370965632306226b88c85f12a34ea5ec8a81` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/compfeat.py` | `chainF/case/frozen/eval/filseg/compfeat.py` | `56483bafa6094f631e19b1b9534a0deebc1a14794b017b36dd22a250dbc7e7f8` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/data/test_filenames.txt` | `chainF/case/frozen/eval/filseg/data/test_filenames.txt` | `1f7bdacc526384386e4ae4fc9d33f0a194cebf36e3634d587eb80e961bf485be` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/evaluate.py` | `chainF/case/frozen/eval/filseg/evaluate.py` | `45964e22d1b707689a9523d58f631d4f70f74f211038165380f60be0f280ee49` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/guard.py` | `chainF/case/frozen/eval/filseg/guard.py` | `881dfdcc29998dc43a178b82d98ae2081de57e8e0919ca03458b37b861a2fa6e` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/instancing.py` | `chainF/case/frozen/eval/filseg/instancing.py` | `5f5676e7fa811d89127bbf51812f7050f9b7cb5c3ad5e836731051f2498c8842` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/maps.py` | `chainF/case/frozen/eval/filseg/maps.py` | `498c5b7c65409d5680a7e33a0c921da5ab300f72ed4c7a78d3c6761d557b17c4` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/msiou.py` | `chainF/case/frozen/eval/filseg/msiou.py` | `71b3d2b1e0cae716933044a2bb8fc897c6faf04eead0d6d7d1c274eb486a906b` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/postproc.py` | `chainF/case/frozen/eval/filseg/postproc.py` | `99fea8fffd3bc88903bee02ea17982fc051f81bc3c6920c72a10a6de84a56dc4` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/report.py` | `chainF/case/frozen/eval/filseg/report.py` | `6ebb4893efc9634e35d1283b7c6b55e99ebfb7514c13978793157e029871cf4e` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/splits.py` | `chainF/case/frozen/eval/filseg/splits.py` | `a010ad4fd4554f743922d8e761231ab5692206516dabffb78d27357fc9487326` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/vendor/__init__.py` | `chainF/case/frozen/eval/filseg/vendor/__init__.py` | `0a25e2882d5da0340857c8a3e8faa1883c050d0f6011e04e1244d4b414ab0778` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/filseg/vendor/miou.py` | `chainF/case/frozen/eval/filseg/vendor/miou.py` | `cdc8ae8666ecde935d06a1936803eb5886e0d2fc2a9f453b189a9683a9ec848e` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/make_time_folds.py` | `chainF/case/frozen/eval/make_time_folds.py` | `ab82ac5b7e5a8b931bac50cd511a76c3f4318f30880fc46dcc3d872a3b26dcd9` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/pq_metric.py` | `chainF/case/frozen/eval/pq_metric.py` | `77e5ec61e1e05ec63a75fb1edf12832f383fb2eff0723aef02f2982f4c7320b4` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/eval/quality_features.py` | `chainF/case/frozen/eval/quality_features.py` | `a52a0a5162ac0ba1c57d6e3e4a862b4e0d1c2747b061efef96f761a19e30b981` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/evaluate.py` | `chainF/case/frozen/evaluate.py` | `79c749f97b3805abe3a2aa2304134315a8d8e65e7bbe20ca77c98726228a72e4` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/evidence/fold_table.json` | `chainF/case/frozen/evidence/fold_table.json` | `2a2451b3d4d9420137ac3e9c6ecd97a7741100ecf35f8accdc0d2140138118f8` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/evidence/mf_inputs.json` | `chainF/case/frozen/evidence/mf_inputs.json` | `00f76fe3df7f5d4a8b7cd2de059f6d77097317b8c421dfdc008d828160cc82ac` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/frozen/evidence/mf_result.json` | `chainF/case/frozen/evidence/mf_result.json` | `a10a2cee5a2137b04e7adcaee1094d4ae473485397e3ac88094dfb9c9284189a` | as run, not re-run here; not copied (over 1 MB) | — |
| `selector_and_repair/case/frozen/evidence/triples_protocol.json` | `chainF/case/frozen/evidence/triples_protocol.json` | `259349a9fe927120ed8d506ae41210dafe22fc062a47cc2ef0558033da046f75` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/frozen/evidence/triples_result.json` | `chainF/case/frozen/evidence/triples_result.json` | `35872f3c0f46abd3880ab8965590f584d2561c84f1cafd93dc8d902b315e66b0` | as run, not re-run here; not copied (over 1 MB) | — |
| `selector_and_repair/case/frozen/protocol.json` | `chainF/case/frozen/protocol.json` | `a4e2cb65d62d3b2176b9f5c280524b60f800b661e9e6d5a15bd5ee32a7f3a57e` | as run, not re-run here | no |
| `selector_and_repair/case/frozen/results/eval/MF_remote_manifest_verification.json` | `chainF/case/frozen/results/eval/MF_remote_manifest_verification.json` | `b626f44759076808db9009ab58cadec66553cdb43c809ff545ddca05ef86527a` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/frozen/revision.py` | `chainF/case/frozen/revision.py` | `357557588aa64d9e1d1fad7f743931757be156227c5d269b2fd1a13907a79feb` | as run, not re-run here | no |
| `selector_and_repair/case/frozen_identity.json` | `chainF/case/frozen_identity.json` | `dfbccb5d64e82663b4432f400869a75e432bc7eca9490b1eca6368e09e31d154` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/levels.py` | `chainF/case/levels.py` | `5d6a02d630d0f0c8184bab2ef9eed2a8330a2985efc74d93bf5d1b33648f99ca` | as run, not re-run here | no |
| `selector_and_repair/case/load_inputs.py` | `chainF/case/load_inputs.py` | `e8dd108b505b0bb95c8f5ed19933aa41e4d3e08d602142cdfd9541406ceab2d5` | as run, not re-run here | no |
| `selector_and_repair/case/mf_features.py` | `chainF/case/mf_features.py` | `4ec5e961c27d696606ba4c1a0239b1dd6e2534edda293e536e5fb8d93ae35a85` | as run, not re-run here | no |
| `selector_and_repair/case/mr.py` | `chainF/case/mr.py` | `55c34767bcd25ae7bc8a064122b1b959aaa33d2949aa7276acdc43551d27bf76` | as run, not re-run here | no |
| `selector_and_repair/case/mt.py` | `chainF/case/mt.py` | `94f348a4047fe2b4a603505287133a0d52d9e5fd511ca826bf1cabab41346dd8` | as run, not re-run here | no |
| `selector_and_repair/case/neigh.py` | `chainF/case/neigh.py` | `cf03f7a18948c9f29728d6b24268d67f7fa784b5d0553cf9b171f099af779297` | as run, not re-run here | no |
| `selector_and_repair/case/pc.py` | `chainF/case/pc.py` | `8bfadb427d89c7d4f73127c8963dd4868fb243cb06c8963d84b353462d0e7721` | as run, not re-run here | no |
| `selector_and_repair/case/pcand.py` | `chainF/case/pcand.py` | `87e96105ccc78adc0cf690547cc3f6f3e0b5e2d61bf899a6f118e7731540e201` | as run, not re-run here | no |
| `selector_and_repair/case/pcand2.py` | `chainF/case/pcand2.py` | `d85f5ff6f2b9e868c8ab172ade4a8068c271c5822c8d191f3b9b5fbbd30c72bc` | as run, not re-run here | no |
| `selector_and_repair/case/pcand_v1_reference.py` | `chainF/case/pcand_v1_reference.py` | `87e96105ccc78adc0cf690547cc3f6f3e0b5e2d61bf899a6f118e7731540e201` | as run, not re-run here | no |
| `selector_and_repair/case/pmerge.py` | `chainF/case/pmerge.py` | `663ba9117d4d4946bf4de987435652113e62f0fbca4577a122526fca43573d14` | as run, not re-run here | no |
| `selector_and_repair/case/pnat.py` | `chainF/case/pnat.py` | `dcfb2ce0f2c8a15c6c541bc935811a185d7fc4f5fc69df5e15eda36d3c37d038` | as run, not re-run here | no |
| `selector_and_repair/case/presel.py` | `chainF/case/presel.py` | `822c8b492949c23911ab8ce1231fd877bb717120109ae40d447fbac6fd49421d` | as run, not re-run here | no |
| `selector_and_repair/case/protocol.json` | `chainF/case/protocol.json` | `165704e53a0e86579d1b0f4d24954a2f369794bed39a83a8709ce4ca8735f1b7` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/pt.py` | `chainF/case/pt.py` | `d0014278e265a90d7257bb714fc51802d8c2d657d8d2d9c839ef0cbe1dbc2df8` | as run, not re-run here | no |
| `selector_and_repair/case/pworker.py` | `chainF/case/pworker.py` | `bdf884ad41e72187014b944a3b77e4729048e9afe75269fb80fd2253d6063165` | as run, not re-run here | no |
| `selector_and_repair/case/refine_ops.py` | `chainF/case/refine_ops.py` | `2fbd926ac78eb6b5aeebe1eb0be944988ba12af6074e784668a4c2b389bd9337` | as run, not re-run here | no |
| `selector_and_repair/case/rules.py` | `chainF/case/rules.py` | `7ec706d730166a52bd557f3e6a935c76e045278f123424c78a30dbceaa6b1fbf` | as run, not re-run here | no |
| `selector_and_repair/case/rulespec.py` | `chainF/case/rulespec.py` | `2291648f567560730766b7d368aef0efa587df8301bb8c82460aa4fd8783458e` | as run, not re-run here | no |
| `selector_and_repair/case/run_tests.py` | `chainF/case/run_tests.py` | `bda9b663de29eca9c09964f65c55354c03850a14c3802a737eb75137a67826da` | as run, not re-run here | no |
| `selector_and_repair/case/spatial.py` | `chainF/case/spatial.py` | `e5333f02d83059ccd2e67e7332f528cef08063fd72da473555eca519f4b773a9` | as run, not re-run here | no |
| `selector_and_repair/case/support/bank_inventory.json` | `chainF/case/support/bank_inventory.json` | `816da28e696fa001016ed459c051a63b115f4967b61bc66172eecc822212bfb6` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/support/confirmation_inputs.json` | `chainF/case/support/confirmation_inputs.json` | `e54c4d27f31dfeee019ae96ff11b0defcd8fd8499e10a276b414ad21b22eb06b` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/support/confirmation_protocol.json` | `chainF/case/support/confirmation_protocol.json` | `7db3e2c6f51dd418ecd8bc33b3c8255cf55627febffed9fff3a4f5d0dbe41e64` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/support/confirmation_worker.py` | `chainF/case/support/confirmation_worker.py` | `4a024da1e85f18cbe6e8ec1ee12ea606eacf5c6d5d1c3a98b411bbedf8c88b05` | as run, not re-run here | an owner e-mail in an input-record assert replaced by <owner-email>; the copied records carry the same placeholder, so the check is unchanged |
| `selector_and_repair/case/support/core.py` | `chainF/case/support/core.py` | `604b28f331515f56404faf04e3e5fd587b094c91b510c0e582e9f35327a80095` | as run, not re-run here | no |
| `selector_and_repair/case/support/instrument_source.py` | `chainF/case/support/instrument_source.py` | `af10e93666331829e4d576426754a81a492353f568489c70d59821360fc4bdd4` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_combined.py` | `chainF/case/tests/test_combined.py` | `7f469dfb1e2e2a0bc06a1909d25b5e945c166b10d08b4303e9a596ba642dad66` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_combined_f.py` | `chainF/case/tests/test_combined_f.py` | `5cd745f6dab6f67ebac72cc633e47a2c02612f5b7bc91d7904b2ae62ee6bc272` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_context_features.py` | `chainF/case/tests/test_context_features.py` | `2d8afd6aa58a71293bb27ad1b7926967a61c78a6cffa112840132fd21117148a` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_counts.py` | `chainF/case/tests/test_counts.py` | `b85947ecb5724a5972f89b397cc497cbc44ad4067c6b70f6f6c6f992a4e47e72` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_levels.py` | `chainF/case/tests/test_levels.py` | `cd373217606b7a5c21594fd556c6e8df440cf6340fb4cb8295435bcdf67f54f3` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_main_flags_f.py` | `chainF/case/tests/test_main_flags_f.py` | `b146676b399b040136e1129b0cef55c71e8222748cfc40fe375de863de217871` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_mf_features.py` | `chainF/case/tests/test_mf_features.py` | `8211324b7a47b1674f00b8d2e75e224f94d3b1f29db31d2048fcd0151726df58` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_mr.py` | `chainF/case/tests/test_mr.py` | `b55267ad1cdc5c513914bd1f75005a66f8d0fd381beeb7bc1ffee7992c0ff28d` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_mt.py` | `chainF/case/tests/test_mt.py` | `884bdaf246a8c391637a5a0ad067a55ec50ae2c10ef8550ebda487a7df630b24` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_mt_levels.py` | `chainF/case/tests/test_mt_levels.py` | `79abf142e60f22c07bfba6b08a171c34653e8e3b79d5aaf96b4f222f48fb666f` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_neigh.py` | `chainF/case/tests/test_neigh.py` | `05d56a7804a7ab94d14baab4ba86332d6d67d37f38dec165ec6f3f254b51e9c8` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_no_w25.py` | `chainF/case/tests/test_no_w25.py` | `c8ad4f6e79075b68be4e6abd3c2fe2c2222d7d86001616ccef6437b0f486b8ea` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pc.py` | `chainF/case/tests/test_pc.py` | `8a1c726613f79d669d74170bed7f1e47b30411e3c0a64cd2b49e7699b88b75bc` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pcand.py` | `chainF/case/tests/test_pcand.py` | `1ca4690e73f828f1b03cbbec7a0621bfe7cd937b51719b6df950589d693095db` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/tests/test_pcand2.py` | `chainF/case/tests/test_pcand2.py` | `cf63804d21334691479edc05379ad959cd98228ead695a9b3b3b7143ff97a1c3` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/tests/test_pcand_refit.py` | `chainF/case/tests/test_pcand_refit.py` | `0476e33986519649f94a760bcd5160083a00f3f297b4577aaaa4a325092dec67` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pmerge.py` | `chainF/case/tests/test_pmerge.py` | `425990500c7d1c042677ae537a97a2bffac3da47cdd4a284db1fb36de5652919` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pnat.py` | `chainF/case/tests/test_pnat.py` | `6e23c8133edb9d925fdd9c7eb6c7dc989b4a8073a8b1fe8af12fb9370c4b2f5d` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_presel.py` | `chainF/case/tests/test_presel.py` | `ae7ae129166b63002e3cd3a1a1d9fc5cbb367825cd96fd7d749ed87582d71b06` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pt.py` | `chainF/case/tests/test_pt.py` | `e99cf2e65df6e4b89928a1756fad9b6d68b56a4e06a80e2ca18cadc8e6075bc2` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_pworker.py` | `chainF/case/tests/test_pworker.py` | `d2fcee6935e24f0448ee59c49c5726ec2f22fc81f865c78ce3dbacc25eb3ff70` | as run, not re-run here | private identifiers in records/comments replaced by placeholders |
| `selector_and_repair/case/tests/test_rulespec.py` | `chainF/case/tests/test_rulespec.py` | `408e6c31c593465bb0af66a408c60d179347917073b77605bfce6000317e072e` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_spatial.py` | `chainF/case/tests/test_spatial.py` | `7543010d01a0e793ccdccadbe8beb2b76c77805b60fc80958829dbd3d2036509` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_unet_swap.py` | `chainF/case/tests/test_unet_swap.py` | `163c53077fc22e69e240a0fb6a4a27f0ce62dcf5392d562a7f2ee34bc9d7ac29` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_worker_levels.py` | `chainF/case/tests/test_worker_levels.py` | `eb9db2934b665cbac2a77b8ea091dca5a71cc2195c116b17ca76e4d958ea2527` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_worker_rule.py` | `chainF/case/tests/test_worker_rule.py` | `07da5f473fecb8fc33a01e888f7a1937584fa25f9c77688d43f767aa385f8c1b` | as run, not re-run here | no |
| `selector_and_repair/case/tests/test_worker_synthetic.py` | `chainF/case/tests/test_worker_synthetic.py` | `4d01ede0f22a5c4736a5560174fb6937ca0093d080f4fffad66e3a126b01b279` | as run, not re-run here | no |
| `selector_and_repair/case/worker.py` | `chainF/case/worker.py` | `97832483ab93647271a97cf4a3f5d7fad215989579e649fbef66b3d36ddf645d` | as run, not re-run here | no |
| `selector_and_repair/csv_worker.py` | `chainF/csv_worker.py` | `ada24af501c219a3e8cc2a7db41f08d928601f3e2fad449ff67edd8f9a8edb07` | as run, not re-run here | no |
| `selector_and_repair/finish.py` | `chainF/finish.py` | `9fb08482ff630a7fbe714070179fe521a4eaef0fde9944a165df43f5e32f3277` | as run, not re-run here | no |
| `selector_and_repair/stage.py` | `chainF/stage.py` | `7f04ff05f94c70a952447d165ae7362ffdd6bbb94a5528ffb26d0eda368b65b7` | as run, not re-run here | one dataset owner replaced by <kaggle-user> (metadata of the staged dataset) |
| `selector_and_repair/render_kernel.py` | `chainF/render_kernel.py` | `c2562aa2f93e78333fc17ad52c105f052b5ad71b626cac6c2d6589eae6647172` | as run, not re-run here | Kaggle user and dataset/kernel owners replaced by <kaggle-user>: a rendered kernel attaches nothing until you set your own |
| `selector_and_repair/paired_vs_ref.py` | `chainF/host/tools/paired_vs_ref.py` | `e9379523ab7f1cde13c38d0a9c55723c324b988745449a245a79f42357bd3f69` | as run, not re-run here | chain folder = this file's folder (was an absolute path) |
| `selector_and_repair/cli_ms125.json` | `chainF/host/stage/ms125/cli.json` | `2d3263bf0ab297f5a56602e95bfa1d6c8093ba112c08d703ea5bfbb43e07cb99` | as run, not re-run here | no |
| `selector_and_repair/kernel_ms125.py` | `chainF/host/state/ms125_kernel.py` | `76e7b1a713072fec9445e4ab8b637d152336a6fc059e7dbf52d8c247522fe946` | as run, not re-run here | the embedded payload (base64 tar.xz of the case files, unscrubbed) removed: the driver reads it from the file named by env CHAIN_PAYLOAD_B64. Its digest table still lists the as-run hashes, so scrubbed files fail that check |

After the copy, the storage folder name of the project was also replaced by `<project>` in these files (paths in records and comments only): `selector_and_repair/case/evidence/fold2_map_provenance.json`, `selector_and_repair/case/frozen/evidence/mf_inputs.json`, `selector_and_repair/case/frozen/evidence/triples_protocol.json`, `selector_and_repair/case/frozen/results/eval/MF_remote_manifest_verification.json`, `selector_and_repair/case/support/bank_inventory.json`, `selector_and_repair/case/support/confirmation_inputs.json`, `selector_and_repair/case/support/confirmation_protocol.json`.
