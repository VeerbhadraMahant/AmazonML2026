# STATUS (shared by all agents; append timestamped lines)
21:09 [A] started score_test.py ctx -> work/test_scored_ctx.parquet
21:10 [B] decoder-stacker started (stage-2 stacker + decoders on val_scored_ctx)
21:10 [C] packager started: docs/README/requirements/zip script
21:13 [C] updated Documentation_template.md, code/.../README.md, requirements.txt (py3.13.7 header); wrote src/make_submission_zip.py (dry-run OK, 26 files, 1.6GB). Placeholders {{...}} left for final numbers + STAGE2_TRAIN_ARGS. Build: cd code/business_entity_resolution/src && python make_submission_zip.py --team NAME
21:14 [A] safety submission submissions/ctx_t091/matching_results.tsv PASS validator (5.73M pairs, singleton 5.70%)
21:15 [B] stage2 CV(2-fold by S1) full feats: tau-dec F0.5=0.9542 (tau=.66) India=.9415 US=.9627; expF decoder 0.9552 (India .9431 US .9632). stage1 baseline same scorer 0.9460. CAVEAT checking: val entity windows miss train-S1 competitors
21:16 [A] running oof_stage1.py -> work/oof_train_scored.parquet (all 119.7M train pairs, 2-fold OOF) + work/test_scored_oof.parquet (avg of fold models). ETA ~21:45
21:18 [A] err analysis val@.91: oracle ceiling 0.9909; blocking recall 97.35%; FP 16.1K (13.6K distractor records w/ no owner); FN in-cand 52.8K
21:27 [B] stage2 CV w/ FULL entity context (train-S1 competitors scored by stage-1) + source feats: baseline(full argmax) 0.9463 | stage2 tau=.64 0.9564 (IN .9443 US .9645) | stage2+expF(floor .4 gain .8) 0.9573 (IN .9459 US .9649). train_oof path ready in stack_stage2.py for A's OOF files.
21:28 [A] READY work/oof_train_scored.parquet (119.7M, OOF 2-fold). test_scored_oof.parquet ETA ~21:36
21:36 [A] READY work/test_scored_oof.parquet (120.8M; avg of fold models). sanity decode tau .91 done
21:37 [A] blocking-miss analysis: 20.2K missed val pairs (India 13.9K), mostly empty-address generic names + phonetic translit; NOT pursuing (needs 40+min re-retrieval on test)
21:38 [B] OOF stage2 (train 237k train-S1s, eval 220k val S1s, all windows full-context): stage1-oof baseline tau=.90 0.9452 (IN .9308 US .9548) | stage2 tau=.66 0.9567 | stage2+expF 0.9576 (IN .9460 US .9653). Val logloss rose at 300 rounds -> retraining w/ 200 rounds (ETA 21:46), then handing off apply_test cmd.
21:40 [B] READY-200: lgbm_stage2.txt/stage2_config.json (OOF CV 0.9571 IN .9455 US .9649, expF floor.4 gain.8). cmd: python stack_stage2.py apply_test OUT.tsv test_scored_oof.parquet. Waiting for APPLY-DONE before big training.
21:41 [A] running apply_test stage2_200 -> submissions/stage2_200/matching_results.tsv
21:44 [A] submissions/stage2_200/matching_results.tsv PASS validator (5.83M pairs, singleton 5.35%) -> candidate for upload
21:51 [B] started train_big (30% train S1s, ES on train-S1 holdout, eval val S1s) -> lgbm_stage2_big.txt / stage2_big_config.json
21:56 [B] BIG stage2 done: 594k train S1s, ES@200 on train-S1 holdout; val S1 CV expF(floor.2 gain.8) 0.9574 (IN .9459 US .9651) vs 200-model 0.9571. READY-BIG cmd: python stack_stage2.py apply_test OUT.tsv test_scored_oof.parquet big. Stage-3 skipped (no time/RAM for OOF stage-2 on 120M rows).
22:04 [B] S1-drop sim (21% non-val S1 dropped): stage2_big expF r=1 0.9569 (vs 0.9574 undropped) -> hypothesis NOT supported; prior_r<1 hurts stage2 (r=.5 0.9546, r=.2 0.9454). Keep r=1. Retrain on dropped set running.
22:08 [B] retrain on S1-dropped regime: sim-eval expF 0.9573 (IN .9458 US .9650) vs big 0.9569 -> FINAL = lgbm_stage2_final.txt + stage2_final_config.json (expF floor.4 gain.8 r=1). cmd: python stack_stage2.py apply_test OUT.tsv test_scored_oof.parquet final. Note test S1 mix India 47%/US 38%/France 15% vs train India 40% -> expect lower LB from mix+France alone.
22:19 [A] output/matching_results.tsv = stage2_final (val .9573). old baseline moved to submissions/baseline_sep25
22:20 [B] started train_xgb (GPU, drop sim 0.21, train_frac .35) + ensemble eval
22:31 [B] XGB(GPU) drop-sim val: xgb alone 0.9621 (IN .9510 US .9695) > final+2xgb .9611 > final+xgb .9605 > lgb final .9573. ENS config = xgb alone, expF floor.2 gain.9. cmd: apply_test OUT test_scored_oof.parquet ens. XGB#2 (depth11 eta.03 bin512 frac.5) launched.
22:39 [A] output/matching_results.tsv = stage2_xgb (GPU XGB stage2, val .9621 test-like). PASS validator
22:59 [B] XGB#2 (depth10 bin512 seed7 frac.45, stopped 22:57 @1941): alone 0.9629 (IN .9521 US .9701); xgb1+xgb2 0.9627; +final 0.9622. vs xgb1 0.9621 -> +0.0008 (<0.001). stage2_ens2_config.json written (xgb2 alone). XGB#3 launched (seed11, offset S1 sample, stop 23:10).
23:11 [B] XGB#3: alone 0.9625; xgb2+xgb3 0.9631 (IN .9523 US .9703) BEST; all3 0.9630. stage2_ens3_config.json = xgb2+xgb3 expF floor.2 gain.8. tag ens3.
23:18 [A] FINAL output/matching_results.tsv = ens3 (xgb2+xgb3, val .9631), PASS validator; rebuilding zip
23:37 [B] continuation of xgb2/xgb3 aborted: launch landed after the 23:24/23:34 wall-clock stops; killed. NO GAIN - ens3 (0.9631) stays final.
