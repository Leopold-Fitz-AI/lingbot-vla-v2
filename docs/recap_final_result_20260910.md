# Fixed 50-task final study

Study: `/dev/shm/recap_confirmatory_20260908_locked_cohorts`

**All frozen success criteria passed.**

- Primary episodes: 6000 (paired, both policies actually executed).
- Null macro success: 92.8667%.
- RECAP macro success: 93.1333%.
- Equal-task macro change: +0.2667 percentage points.
- Paired 95% bootstrap CI: [+0.1000, +0.4333] pp.
- Exact two-sided paired p: 0.0078125.
- Improved/regressed pairs: 8/0.
- Unadapted regression guards passed: True.
- Adapted tasks: click_bell.

This is a fixed-suite mean claim, NOT improvement on every task or unseen-task generalization.
Historical 4496/5000 (89.92%) is inventory only, not this experiment's comparator.
No production registry promotion. No extra samples or outcome-dependent retries.

## All task contributions

| Task | Adapted | Null | RECAP | Delta (pp) | Suite contribution (pp) | Wins / losses | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| adjust_bottle | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| beat_block_hammer | no | 56/60 | 56/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| blocks_ranking_rgb | no | 53/60 | 53/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| blocks_ranking_size | no | 45/60 | 45/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| click_alarmclock | no | 57/60 | 57/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| click_bell | yes | 50/60 | 58/60 | +13.333 | +0.2667 | 8 / 0 | 0.390625 |
| dump_bin_bigbin | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| grab_roller | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| handover_block | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| handover_mic | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| hanging_mug | no | 33/60 | 33/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| lift_pot | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| move_can_pot | no | 59/60 | 59/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| move_pillbottle_pad | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| move_playingcard_away | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| move_stapler_pad | no | 52/60 | 52/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| open_laptop | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| open_microwave | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| pick_diverse_bottles | no | 46/60 | 46/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| pick_dual_bottles | no | 54/60 | 54/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_a2b_left | no | 54/60 | 54/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_a2b_right | no | 52/60 | 52/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_bread_basket | no | 57/60 | 57/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_bread_skillet | no | 49/60 | 49/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_burger_fries | no | 59/60 | 59/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_can_basket | no | 48/60 | 48/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_cans_plasticbox | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_container_plate | no | 59/60 | 59/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_dual_shoes | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_empty_cup | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_fan | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_mouse_pad | no | 56/60 | 56/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_object_basket | no | 53/60 | 53/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_object_scale | no | 57/60 | 57/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_object_stand | no | 59/60 | 59/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_phone_stand | no | 55/60 | 55/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| place_shoe | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| press_stapler | no | 55/60 | 55/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| put_bottles_dustbin | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| put_object_cabinet | no | 56/60 | 56/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| rotate_qrcode | no | 59/60 | 59/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| scan_object | no | 54/60 | 54/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| shake_bottle | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| shake_bottle_horizontally | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| stack_blocks_three | no | 58/60 | 58/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| stack_blocks_two | no | 60/60 | 60/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| stack_bowls_three | no | 48/60 | 48/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| stack_bowls_two | no | 57/60 | 57/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| stamp_seal | no | 55/60 | 55/60 | +0.000 | +0.0000 | 0 / 0 | 1 |
| turn_switch | no | 51/60 | 51/60 | +0.000 | +0.0000 | 0 / 0 | 1 |

## Secondary signed-Negative diagnostic

- click_bell: Negative 41/60; Null 50/60. Not a primary endpoint or selection rule.
