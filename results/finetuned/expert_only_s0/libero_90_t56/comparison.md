| suite | task | instruction | baseline success (95% CI) | finetuned success (95% CI) | mean steps |
|---|---|---|---|---|---|
| libero_90 | 56 | pick up the butter and put it in the tray | 0/50 = 0% (0%-7%) | 45/50 = 90% (79%-96%) | 258 |
| **all** | | | **0/50 = 0% (0%-7%)** |  | |
| **all** | | |  | **45/50 = 90% (79%-96%)** | |

**Paired (50 identical initial states):** baseline 0% -> finetuned 90%, diff +90% (bootstrap 95% CI +80% to +98%); discordant pairs: only-baseline 0, only-finetuned 45; exact McNemar p = 5.68e-14
